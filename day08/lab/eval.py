"""
eval.py — Sprint 4: Evaluation & Scorecard
==========================================
Mục tiêu Sprint 4 (60 phút):
  - Chạy 10 test questions qua pipeline
  - Chấm điểm theo 4 metrics: Faithfulness, Relevance, Context Recall, Completeness
  - So sánh baseline vs variant
  - Ghi kết quả ra scorecard

Definition of Done Sprint 4:
  ✓ Demo chạy end-to-end (index → retrieve → answer → score)
  ✓ Scorecard trước và sau tuning
  ✓ A/B comparison: baseline vs variant với giải thích vì sao variant tốt hơn

A/B Rule (từ slide):
  Chỉ đổi MỘT biến mỗi lần để biết điều gì thực sự tạo ra cải thiện.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from rag_answer import rag_answer

# =============================================================================
# CẤU HÌNH
# =============================================================================

TEST_QUESTIONS_PATH = Path(__file__).parent / "data" / "test_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Chấm điểm tự động bằng LLM hay thủ công
# "llm"    → gọi LLM để chấm (cần API key, chậm hơn nhưng nhất quán)
# "manual" → trả về None, yêu cầu điền tay vào CSV sau
SCORING_MODE = os.getenv("SCORING_MODE", "llm")

BASELINE_CONFIG = {
    "retrieval_mode": "dense",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": False,
    "label": "baseline_dense",
}

VARIANT_CONFIG = {
    "retrieval_mode": "hybrid",
    "top_k_search": 10,
    "top_k_select": 3,
    "use_rerank": True,
    "label": "variant_hybrid_rerank",
}

# Singleton LLM client
_openai_client = None


# =============================================================================
# LLM JUDGE — helper chung
# =============================================================================

def _call_judge_llm(prompt: str) -> str:
    """Gọi LLM để chấm điểm. Dùng temperature=0 để output ổn định."""
    global _openai_client

    backend = os.getenv("LLM_BACKEND", "openai").lower()

    if backend == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text.strip()

    # Default: OpenAI
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    response = _openai_client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    """Parse JSON từ LLM output, bỏ qua markdown fences nếu có."""
    import re
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"score": None, "reason": raw[:200]}


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def score_faithfulness(
    answer: str,
    chunks_used: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Faithfulness (1-5): Câu trả lời có bám đúng chứng cứ đã retrieve không?

    5 = hoàn toàn grounded trong retrieved context
    1 = phần lớn thông tin không có trong retrieved chunks (model hallucinate)

    Nếu answer là abstain ("Không đủ dữ liệu...") → auto score 5 (không bịa).
    """
    # Abstain answer luôn faithful
    abstain_markers = ["không đủ dữ liệu", "do not know", "i don't know", "cannot find"]
    if any(m in answer.lower() for m in abstain_markers):
        return {"score": 5, "notes": "Model correctly abstained — faithful by definition"}

    if SCORING_MODE == "manual" or not chunks_used:
        return {"score": None, "notes": "Manual scoring required"}

    context = "\n\n".join(
        f"[{i+1}] {c.get('text', '')[:400]}"
        for i, c in enumerate(chunks_used)
    )

    prompt = f"""You are an evaluation judge for a RAG system.

Retrieved context:
{context}

Model answer:
{answer}

Task: Rate FAITHFULNESS on a scale of 1-5.
- 5: Every claim in the answer is directly supported by the retrieved context above.
- 4: Almost entirely grounded; one minor detail is uncertain.
- 3: Most claims grounded; some info may come from model's prior knowledge.
- 2: Several claims not found in the retrieved context.
- 1: Answer is mostly hallucinated; not grounded in the retrieved context.

Output ONLY valid JSON (no markdown fences):
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        raw = _call_judge_llm(prompt)
        result = _parse_judge_json(raw)
        return {
            "score": result.get("score"),
            "notes": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": None, "notes": f"Judge error: {e}"}


def score_answer_relevance(
    query: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Answer Relevance (1-5): Answer có trả lời đúng câu hỏi người dùng hỏi không?

    5 = trả lời trực tiếp và đầy đủ câu hỏi
    1 = hoàn toàn lạc đề

    Abstain answer → score 3 (không sai nhưng cũng không đầy đủ).
    """
    abstain_markers = ["không đủ dữ liệu", "do not know", "i don't know", "cannot find"]
    if any(m in answer.lower() for m in abstain_markers):
        return {"score": 3, "notes": "Model abstained — relevant but not informative"}

    if SCORING_MODE == "manual":
        return {"score": None, "notes": "Manual scoring required"}

    prompt = f"""You are an evaluation judge for a RAG system.

User question: {query}

Model answer: {answer}

Task: Rate ANSWER RELEVANCE on a scale of 1-5.
- 5: Directly and fully addresses the user's question.
- 4: Addresses the question but misses minor sub-points.
- 3: Partially relevant; the core question is acknowledged but not fully answered.
- 2: Tangentially related; the answer drifts off-topic.
- 1: Completely off-topic or does not address the question at all.

Output ONLY valid JSON (no markdown fences):
{{"score": <integer 1-5>, "reason": "<one sentence>"}}"""

    try:
        raw = _call_judge_llm(prompt)
        result = _parse_judge_json(raw)
        return {
            "score": result.get("score"),
            "notes": result.get("reason", ""),
        }
    except Exception as e:
        return {"score": None, "notes": f"Judge error: {e}"}


def score_context_recall(
    chunks_used: List[Dict[str, Any]],
    expected_sources: List[str],
) -> Dict[str, Any]:
    """
    Context Recall (1-5): Retriever có mang về đủ evidence cần thiết không?

    Tính theo partial-match: tên file (không cần full path).
    recall = found / total_expected → convert sang thang 1-5.
    """
    if not expected_sources:
        return {"score": None, "recall": None, "notes": "No expected sources defined"}

    retrieved_sources = {
        c.get("metadata", {}).get("source", "")
        for c in chunks_used
    }

    found = 0
    missing = []
    for expected in expected_sources:
        # Partial match trên tên file (bỏ extension và path prefix)
        expected_stem = Path(expected).stem.lower()
        matched = any(expected_stem in Path(r).stem.lower() for r in retrieved_sources)
        if matched:
            found += 1
        else:
            missing.append(expected)

    recall = found / len(expected_sources)

    # Convert recall [0,1] → score [1,5]
    # 1.0 → 5,  0.75 → 4,  0.5 → 3,  0.25 → 2,  0.0 → 1
    score = max(1, round(recall * 4) + 1)

    return {
        "score": score,
        "recall": round(recall, 3),
        "found": found,
        "missing": missing,
        "notes": (
            f"Retrieved {found}/{len(expected_sources)} expected sources"
            + (f". Missing: {missing}" if missing else "")
        ),
    }


def score_completeness(
    query: str,
    answer: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Completeness (1-5): Answer có thiếu điều kiện ngoại lệ hoặc bước quan trọng không?

    5 = bao gồm đủ tất cả điểm quan trọng trong expected_answer
    1 = thiếu phần lớn nội dung cốt lõi

    Nếu không có expected_answer → skip (score = None).
    """
    if not expected_answer or expected_answer.strip() == "":
        return {"score": None, "notes": "No expected answer provided — skipped"}

    abstain_markers = ["không đủ dữ liệu", "do not know", "i don't know", "cannot find"]
    if any(m in answer.lower() for m in abstain_markers):
        return {"score": 1, "notes": "Model abstained — answer is incomplete"}

    if SCORING_MODE == "manual":
        return {"score": None, "notes": "Manual scoring required"}

    prompt = f"""You are an evaluation judge for a RAG system.

User question: {query}

Expected answer (ground truth key points):
{expected_answer}

Model answer:
{answer}

Task: Rate COMPLETENESS on a scale of 1-5.
- 5: The model answer covers ALL key points from the expected answer.
- 4: Covers most key points; missing one minor detail.
- 3: Covers the main point but omits some important conditions or steps.
- 2: Covers less than half of the key points.
- 1: Misses most or all key points from the expected answer.

Output ONLY valid JSON (no markdown fences):
{{"score": <integer 1-5>, "missing_points": ["<point1>", "..."], "reason": "<one sentence>"}}"""

    try:
        raw = _call_judge_llm(prompt)
        result = _parse_judge_json(raw)
        missing_pts = result.get("missing_points", [])
        notes = result.get("reason", "")
        if missing_pts:
            notes += f" Missing: {missing_pts}"
        return {
            "score": result.get("score"),
            "notes": notes,
        }
    except Exception as e:
        return {"score": None, "notes": f"Judge error: {e}"}


# =============================================================================
# SCORECARD RUNNER
# =============================================================================

def run_scorecard(
    config: Dict[str, Any],
    test_questions: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chạy toàn bộ test questions qua pipeline và chấm 4 metrics.

    Returns:
        List scorecard rows, mỗi row là một câu hỏi với đầy đủ scores.
    """
    if test_questions is None:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)

    results = []
    label = config.get("label", "unnamed")
    pipeline_config = {k: v for k, v in config.items() if k != "label"}

    print(f"\n{'='*70}")
    print(f"Scorecard: {label}")
    print(f"Config: {pipeline_config}")
    print(f"Scoring mode: {SCORING_MODE}")
    print("="*70)

    for idx, q in enumerate(test_questions, 1):
        question_id = q.get("id", f"Q{idx:02d}")
        query = q["question"]
        expected_answer = q.get("expected_answer", "")
        expected_sources = q.get("expected_sources", [])
        category = q.get("category", "general")

        if verbose:
            print(f"\n[{question_id}] {query}")

        # --- Gọi RAG pipeline ---
        try:
            result = rag_answer(query=query, verbose=False, **pipeline_config)
            answer = result["answer"]
            chunks_used = result["chunks_used"]
        except NotImplementedError:
            answer = "PIPELINE_NOT_IMPLEMENTED"
            chunks_used = []
        except Exception as e:
            answer = f"ERROR: {type(e).__name__}: {e}"
            chunks_used = []

        # --- Chấm 4 metrics ---
        faith = score_faithfulness(answer, chunks_used)
        relevance = score_answer_relevance(query, answer)
        recall = score_context_recall(chunks_used, expected_sources)
        complete = score_completeness(query, answer, expected_answer)

        row = {
            "id": question_id,
            "category": category,
            "query": query,
            "answer": answer,
            "expected_answer": expected_answer,
            # Scores
            "faithfulness": faith["score"],
            "faithfulness_notes": faith.get("notes", ""),
            "relevance": relevance["score"],
            "relevance_notes": relevance.get("notes", ""),
            "context_recall": recall["score"],
            "context_recall_raw": recall.get("recall"),
            "context_recall_notes": recall.get("notes", ""),
            "completeness": complete["score"],
            "completeness_notes": complete.get("notes", ""),
            # Meta
            "config_label": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        results.append(row)

        if verbose:
            scores_str = (
                f"Faith={faith['score']} | Rel={relevance['score']} | "
                f"Recall={recall['score']} | Complete={complete['score']}"
            )
            print(f"  Answer  : {answer[:120]}{'...' if len(answer) > 120 else ''}")
            print(f"  Scores  : {scores_str}")

    # --- Print averages ---
    _print_averages(results, label)

    return results


def _print_averages(results: List[Dict], label: str) -> None:
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    print(f"\n{'─'*50}")
    print(f"Summary — {label}")
    print(f"{'─'*50}")
    for metric in metrics:
        scores = [r[metric] for r in results if r.get(metric) is not None]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {metric:<20}: {avg:.2f}/5  (n={len(scores)})")
        else:
            print(f"  {metric:<20}: N/A (chưa chấm)")


# =============================================================================
# A/B COMPARISON
# =============================================================================

def compare_ab(
    baseline_results: List[Dict],
    variant_results: List[Dict],
    output_csv: Optional[str] = None,
) -> None:
    """
    So sánh baseline vs variant theo metric và theo từng câu.

    In bảng delta và highlight câu nào variant tốt / kém hơn.
    Lưu kết quả ra CSV nếu output_csv được cung cấp.
    """
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]

    print(f"\n{'='*70}")
    print("A/B Comparison: Baseline vs Variant")
    print("="*70)
    print(f"{'Metric':<22} {'Baseline':>10} {'Variant':>10} {'Delta':>8} {'Better?':>10}")
    print("─" * 65)

    for metric in metrics:
        b_scores = [r[metric] for r in baseline_results if r.get(metric) is not None]
        v_scores = [r[metric] for r in variant_results if r.get(metric) is not None]
        b_avg = sum(b_scores) / len(b_scores) if b_scores else None
        v_avg = sum(v_scores) / len(v_scores) if v_scores else None
        delta = (v_avg - b_avg) if (b_avg is not None and v_avg is not None) else None

        b_str = f"{b_avg:.2f}" if b_avg is not None else "N/A"
        v_str = f"{v_avg:.2f}" if v_avg is not None else "N/A"
        d_str = f"{delta:+.2f}" if delta is not None else "N/A"
        better = ("✓ Variant" if delta and delta > 0.05
                  else ("✗ Baseline" if delta and delta < -0.05
                        else "≈ Tie"))

        print(f"{metric:<22} {b_str:>10} {v_str:>10} {d_str:>8} {better:>10}")

    # --- Per-question ---
    print(f"\n{'─'*65}")
    print(f"{'ID':<8} {'Category':<14} {'Baseline':>12} {'Variant':>12} {'Winner':<12}")
    print("─" * 65)

    b_by_id = {r["id"]: r for r in baseline_results}
    for v_row in variant_results:
        qid = v_row["id"]
        b_row = b_by_id.get(qid, {})

        b_total = sum(b_row.get(m) or 0 for m in metrics)
        v_total = sum(v_row.get(m) or 0 for m in metrics)
        b_str = f"{b_total}/{len(metrics)*5}"
        v_str = f"{v_total}/{len(metrics)*5}"

        if v_total > b_total:
            winner = "✓ Variant"
        elif b_total > v_total:
            winner = "✗ Baseline"
        else:
            winner = "≈ Tie"

        print(f"{qid:<8} {v_row.get('category',''):<14} {b_str:>12} {v_str:>12} {winner:<12}")

    # --- CSV export ---
    if output_csv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / output_csv
        combined = baseline_results + variant_results
        if combined:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=combined[0].keys())
                writer.writeheader()
                writer.writerows(combined)
            print(f"\nKết quả đã lưu: {csv_path}")


# =============================================================================
# MARKDOWN REPORT GENERATOR
# =============================================================================

def generate_scorecard_summary(results: List[Dict], label: str) -> str:
    """Tạo báo cáo scorecard dạng markdown, sẵn sàng để commit vào repo."""
    metrics = ["faithfulness", "relevance", "context_recall", "completeness"]
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r.get(metric) is not None]
        averages[metric] = sum(scores) / len(scores) if scores else None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    b_label = results[0].get("config_label", label) if results else label

    md = f"""# Scorecard: {b_label}

**Generated:** {timestamp}  
**Questions evaluated:** {len(results)}  
**Scoring mode:** {SCORING_MODE}

## Average Scores

| Metric | Score (out of 5) |
|--------|-----------------|
"""
    for metric in metrics:
        avg = averages[metric]
        bar = "█" * int(round(avg or 0)) + "░" * (5 - int(round(avg or 0)))
        avg_str = f"{avg:.2f} {bar}" if avg is not None else "N/A"
        md += f"| {metric.replace('_', ' ').title()} | {avg_str} |\n"

    md += f"""
## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Answer Preview |
|----|----------|:--------:|:--------:|:------:|:--------:|----------------|
"""
    for r in results:
        preview = r.get("answer", "")[:60].replace("|", "｜")
        md += (
            f"| {r['id']} | {r['category']} "
            f"| {r.get('faithfulness','–')} "
            f"| {r.get('relevance','–')} "
            f"| {r.get('context_recall','–')} "
            f"| {r.get('completeness','–')} "
            f"| {preview}… |\n"
        )

    md += "\n## Notes\n\n"
    md += "<!-- Add qualitative observations here -->\n\n"
    md += "### Questions where model abstained\n"
    abstain_ids = [
        r["id"] for r in results
        if any(m in r.get("answer", "").lower()
               for m in ["không đủ dữ liệu", "do not know", "cannot find"])
    ]
    if abstain_ids:
        md += ", ".join(abstain_ids) + "\n"
    else:
        md += "None\n"

    return md


def save_results_csv(results: List[Dict], filename: str) -> Path:
    """Lưu scorecard results ra CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    if results:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"CSV saved: {path}")
    return path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sprint 4: Evaluation & Scorecard")
    print("=" * 60)

    # --- Load test questions ---
    print(f"\nLoading: {TEST_QUESTIONS_PATH}")
    try:
        with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"Tìm thấy {len(test_questions)} câu hỏi:")
        for q in test_questions[:3]:
            print(f"  [{q.get('id','?')}] {q['question'][:70]} ({q.get('category','')})")
        if len(test_questions) > 3:
            print(f"  ... và {len(test_questions)-3} câu nữa")
    except FileNotFoundError:
        print(f"⚠ Không tìm thấy {TEST_QUESTIONS_PATH}")
        print("  Hãy tạo file data/test_questions.json theo format:")
        print('  [{"id":"Q01","question":"...","expected_answer":"...","expected_sources":["..."],"category":"..."}]')
        test_questions = []

    if not test_questions:
        exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Baseline ---
    print("\n--- Chạy Baseline ---")
    baseline_results = run_scorecard(
        config=BASELINE_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )
    save_results_csv(baseline_results, "scorecard_baseline.csv")
    baseline_md = generate_scorecard_summary(baseline_results, "baseline_dense")
    (RESULTS_DIR / "scorecard_baseline.md").write_text(baseline_md, encoding="utf-8")
    print(f"Scorecard markdown: {RESULTS_DIR / 'scorecard_baseline.md'}")

    # --- Variant ---
    print("\n--- Chạy Variant ---")
    variant_results = run_scorecard(
        config=VARIANT_CONFIG,
        test_questions=test_questions,
        verbose=True,
    )
    save_results_csv(variant_results, "scorecard_variant.csv")
    variant_md = generate_scorecard_summary(variant_results, VARIANT_CONFIG["label"])
    (RESULTS_DIR / "scorecard_variant.md").write_text(variant_md, encoding="utf-8")
    print(f"Scorecard markdown: {RESULTS_DIR / 'scorecard_variant.md'}")

    # --- A/B Comparison ---
    print("\n--- A/B Comparison ---")
    compare_ab(
        baseline_results,
        variant_results,
        output_csv="ab_comparison.csv",
    )

    print("\n✓ Sprint 4 hoàn thành!")
    print(f"  Kết quả trong: {RESULTS_DIR}/")
    print("  Bước tiếp theo: Điền nhận xét vào scorecard_*.md và cập nhật tuning-log.md")