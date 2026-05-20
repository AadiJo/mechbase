import argparse
import json
from pathlib import Path

from app.rag.config import get_settings
from app.rag.search import search


DEFAULT_QUERIES = [
    {"query": "multi ball shooter", "expected": []},
    {"query": "drum shooter", "expected": []},
    {"query": "multi lane shooter", "expected": []},
    {"query": "floor intake", "expected": []},
    {"query": "indexer", "expected": []},
    {"query": "climber", "expected": []},
    {"query": "coral end effector", "expected": []},
    {"query": "algae intake", "expected": []},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recall-oriented search evals.")
    parser.add_argument("--eval-file", default=None, help="JSON file with [{query, expected:[{source_pdf,page}]}].")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    cases = json.loads(Path(args.eval_file).read_text()) if args.eval_file else DEFAULT_QUERIES
    for case in cases:
        response = search(case["query"], top_k=args.top_k, debug=True, settings=get_settings())
        got = {(result.source_pdf, result.page) for result in response.results}
        expected = {(item["source_pdf"], item["page"]) for item in case.get("expected", [])}
        found = len(got & expected)
        total = len(expected)
        print(f"{case['query']}: recall@{args.top_k}={found}/{total}")
        for idx, result in enumerate(response.results[:5], start=1):
            print(f"  {idx}. {result.source_pdf} p{result.page} {result.modality} score={result.score:.3f}")


if __name__ == "__main__":
    main()
