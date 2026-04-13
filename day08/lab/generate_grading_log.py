"""
Generate grading_run.json from grading_questions.json
Run: python generate_grading_log.py
"""

import json
from datetime import datetime
from pathlib import Path
from rag_answer import rag_answer

GRADING_QUESTIONS_PATH = Path("logs/grading_questions.json")
OUTPUT_PATH = Path("logs/grading_run.json")

def generate_grading_log():
    """Load grading_questions.json, run pipeline, save grading_run.json"""
    
    # Load grading questions
    with open(GRADING_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    log = []
    
    print(f"\n{'='*60}")
    print("Running grading_questions through RAG pipeline")
    print(f"{'='*60}\n")
    
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        
        print(f"[{i}/{len(questions)}] {qid}: {question[:60]}...")
        
        try:
            # Run pipeline with hybrid + rerank (best config)
            result = rag_answer(
                query=question,
                retrieval_mode="hybrid",
                top_k_search=10,
                top_k_select=3,
                use_rerank=True,
                verbose=False,
            )
            
            entry = {
                "id": qid,
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "chunks_retrieved": len(result["chunks_used"]),
                "retrieval_mode": result["config"]["retrieval_mode"],
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:80]}")
            entry = {
                "id": qid,
                "question": question,
                "answer": f"PIPELINE_ERROR: {str(e)[:200]}",
                "sources": [],
                "chunks_retrieved": 0,
                "retrieval_mode": "hybrid",
                "timestamp": datetime.now().isoformat(),
            }
        
        log.append(entry)
        print(f"  ✓ Done")
    
    # Save to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Grading log saved: {OUTPUT_PATH}")
    print(f"  Total: {len(log)} questions")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    generate_grading_log()
