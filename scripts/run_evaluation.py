#!/usr/bin/env python3
"""
Run the 5 evaluation questions via the API and display responses.
Assumes the app container is running at localhost:8000.
"""
 
import asyncio
import sys
 
import httpx
 
BASE_URL = "http://localhost:8000"
 
# Delay between questions (seconds) — respects free tier RPM limits
QUESTION_DELAY = 15
 
EVALUATION_QUESTIONS = [
    (
        "Q1",
        "Qual é o mecanismo central proposto no paper Attention Is All You Need "
        "e como ele se diferencia de RNNs?",
    ),
    (
        "Q2",
        "Como o RAG combina recuperação e geração? "
        "Quais são suas limitações apontadas pelos autores?",
    ),
    (
        "Q3",
        "Compare a abordagem do ReAct com a do Toolformer para uso de ferramentas em LLMs.",
    ),
    (
        "Q4",
        "Qual paper você considera mais relevante para construir um agente com uso de "
        "ferramentas externas? Justifique com base nos textos.",
    ),
    (
        "Q5",
        "Faça um resumo executivo dos 5 papers em no máximo 5 bullet points cada.",
    ),
]
 
 
async def run_evaluation() -> None:
    print("=" * 80)
    print("🧠  ML Paper Analysis Assistant — Evaluation Run")
    print("=" * 80)
 
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Health check
        try:
            health = await client.get(f"{BASE_URL}/health")
            health.raise_for_status()
            print(f"\n✅ API is healthy at {BASE_URL}\n")
        except Exception as exc:
            print(f"\n❌ API is not reachable at {BASE_URL}")
            print(f"   Error: {exc}")
            print("   Make sure 'make setup' has been run and containers are up.")
            sys.exit(1)
 
        # Create a new thread for each question (no shared context)
        errors = 0
        for i, (label, question) in enumerate(EVALUATION_QUESTIONS):
            # Pause between questions to respect RPM limits
            if i > 0:
                print(f"\n⏸  Waiting {QUESTION_DELAY}s before next question...")
                await asyncio.sleep(QUESTION_DELAY)
 
            print(f"\n{'─' * 80}")
            print(f"{label}: {question}")
            print("─" * 80)
            print("⏳ Waiting for response...\n")
 
            try:
                # Fresh thread per question
                t_resp = await client.post(f"{BASE_URL}/threads")
                t_resp.raise_for_status()
                thread_id = t_resp.json()["thread_id"]
 
                resp = await client.post(
                    f"{BASE_URL}/threads/{thread_id}/messages",
                    json={"content": question},
                )
 
                if resp.status_code != 200:
                    print(f"  ✗ ERROR {resp.status_code}: {resp.text}")
                    errors += 1
                    continue
 
                answer = resp.json()["response"]
                print(f"Answer:\n{answer}\n")
 
            except Exception as exc:
                print(f"  ✗ Exception: {exc}")
                errors += 1
 
    print("\n" + "=" * 80)
    if errors == 0:
        print("✅  Evaluation complete — all questions answered.")
    else:
        print(f"⚠️   Evaluation complete — {errors}/{len(EVALUATION_QUESTIONS)} questions failed.")
    print("=" * 80)
 
 
if __name__ == "__main__":
    asyncio.run(run_evaluation())
 