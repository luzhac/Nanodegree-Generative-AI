


import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import llm_client
import rag_client
import ragas_evaluator


def load_questions(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(line)
    return questions


def aggregate_metrics(results: List[Dict[str, object]]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in results:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

    averages = {}
    for key, total in totals.items():
        if counts.get(key):
            averages[key] = total / counts[key]
    return averages


def run_batch_eval(
    dataset_path: str,
    chroma_dir: str,
    collection_name: str,
    openai_key: str,
    model: str,
    n_results: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    questions = load_questions(dataset_path)
    if not questions:
        raise ValueError("Dataset file is empty or contains only whitespace.")

     
    os.environ["OPENAI_API_KEY"] = openai_key

    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        raise RuntimeError(f"Failed to initialize RAG system: {error}")

    results: List[Dict[str, object]] = []
    for idx, question in enumerate(questions, start=1):
        docs_result = rag_client.retrieve_documents(collection, question, n_results=n_results)
        documents = []
        metadatas = []
        if docs_result and docs_result.get("documents"):
            documents = docs_result["documents"][0]
            metadatas = docs_result["metadatas"][0]

        context = rag_client.format_context(documents, metadatas)
        answer = llm_client.generate_response(
            openai_key,
            question,
            context,
            conversation_history=[],
            model=model,
        )

        metrics = ragas_evaluator.evaluate_response_quality(
            question,
            answer,
            documents,
            openai_api_key=openai_key,
        )

        results.append(
            {
                "index": idx,
                "question": question,
                "answer": answer,
                "metrics": metrics,
            }
        )

    return results, aggregate_metrics(results)


def write_outputs(results: List[Dict[str, object]], summary: Dict[str, float], output_prefix: str) -> None:
    json_path = f"{output_prefix}.json"
    csv_path = f"{output_prefix}.csv"

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"results": results, "summary": summary}, handle, indent=2)

    metric_keys = sorted({k for r in results for k in (r.get("metrics") or {}).keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "question", "answer"] + metric_keys)
        for row in results:
            metrics = row.get("metrics") or {}
            writer.writerow(
                [row.get("index"), row.get("question"), row.get("answer")]
                + [metrics.get(k) for k in metric_keys]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch RAG evaluation and output metrics.")
    parser.add_argument("--dataset", default="evaluation_dataset.txt", help="Path to dataset file.")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory.")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="ChromaDB collection name.")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model for generation.")
    parser.add_argument("--n-results", type=int, default=3, help="Number of documents to retrieve.")
    parser.add_argument("--output-prefix", default="batch_eval_results", help="Output filename prefix.")

    args = parser.parse_args()

    if not args.openai_key:
        raise SystemExit("OpenAI API key not provided. Set OPENAI_API_KEY or use --openai-key.")

    results, summary = run_batch_eval(
        dataset_path=args.dataset,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        openai_key=args.openai_key,
        model=args.model,
        n_results=args.n_results,
    )
    write_outputs(results, summary, args.output_prefix)

    print("Batch evaluation complete.")
    print(f"Wrote {len(results)} rows.")
    print(f"Summary metrics: {summary}")


if __name__ == "__main__":
    main()
