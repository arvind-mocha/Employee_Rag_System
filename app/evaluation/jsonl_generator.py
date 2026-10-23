"""CLI for generating reviewable evaluation JSONL from current RAG data."""

import argparse
import json
from pathlib import Path

from app.api.llm import LLMProvider
from app.evaluation.dataset import write_dataset
from app.evaluation.engine import generate_ground_truth_records
from app.evaluation.models import RetrievalConfiguration


def _read_questions(path: Path) -> list[str]:
    questions = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid questions JSONL at line {line_number}"
                ) from error
            question = record.get("question") if isinstance(record, dict) else None
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"line {line_number} must contain a non-empty question"
                )
            questions.append(question)
    if not questions:
        raise ValueError("questions JSONL is empty")
    return questions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an auditable JSONL ground-truth snapshot using current "
            "retrieval results."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument(
        "--reference-provider",
        choices=[provider.value for provider in LLMProvider],
        required=True,
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--semantic-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keyword-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--query-rewriting",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--reranking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--source-diversity",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    retrieval = RetrievalConfiguration(
        semantic_search=args.semantic_search,
        keyword_search=args.keyword_search,
        query_rewriting=args.query_rewriting,
        reranking=args.reranking,
        source_diversity=args.source_diversity,
    )
    records = generate_ground_truth_records(
        _read_questions(args.questions),
        LLMProvider(args.reference_provider),
        retrieval,
    )
    destination = write_dataset(
        args.dataset,
        records,
        directory=args.output_directory,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(records)} records to {destination}")


if __name__ == "__main__":
    main()
