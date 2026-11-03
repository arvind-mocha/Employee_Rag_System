"""Dataset library and explicitly selected RAG evaluation operations."""

from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
import logging
from typing import Annotated
from uuid import uuid4

from fastapi import File, HTTPException, Query, Response, UploadFile, status

from app.api import config
from app.api.llm_usage import summarize_llm_usage
from app.api.services.provider_configuration import validate_evaluation_providers
from app.api.schemas.evaluation import (
    DeleteEvaluationDatasetResponse,
    EvaluationDatasetList,
    EvaluationDatasetPreview,
    EvaluationSchemaReference,
    RunEvaluationRequest,
    RunEvaluationResponse,
    UploadEvaluationDatasetResponse,
)
from app.evaluation.dataset import (
    EvaluationDatasetError,
    dataset_metadata,
    dataset_path,
    delete_dataset,
    list_datasets,
    parse_dataset_jsonl,
    store_uploaded_dataset,
    validate_upload_filename,
)
from app.evaluation.engine import _summary, run_evaluation
from app.evaluation.models import EvaluationCategory, EvaluationRun, GroundTruthRecord


logger = logging.getLogger(__name__)


def _dataset_request_error(error: Exception) -> HTTPException:
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    return HTTPException(status_code=422, detail=str(error))


def list_evaluation_datasets() -> EvaluationDatasetList:
    return EvaluationDatasetList(datasets=list_datasets())


def get_evaluation_example() -> Response:
    """Serve the tracked reference verbatim for both preview and download."""
    source = config.PROJECT_ROOT / "examples" / "evaluation_dataset.example.jsonl"
    return Response(
        content=source.read_bytes(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": 'attachment; filename="evaluation_dataset.example.jsonl"',
            "Cache-Control": "no-cache",
        },
    )


def get_evaluation_schema() -> EvaluationSchemaReference:
    return EvaluationSchemaReference(
        maximum_file_size_bytes=config.MAX_EVALUATION_DATASET_SIZE,
        maximum_records=config.MAX_EVALUATION_RECORDS,
        maximum_upload_files=config.MAX_EVALUATION_UPLOAD_FILES,
        categories=[category.value for category in EvaluationCategory],
        record_schema=GroundTruthRecord.model_json_schema(),
    )


def preview_evaluation_dataset(
    dataset_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EvaluationDatasetPreview:
    try:
        metadata = dataset_metadata(dataset_id)
        records = []
        if metadata["validation_status"] == "valid":
            records = parse_dataset_jsonl(dataset_path(dataset_id).read_bytes())
        return EvaluationDatasetPreview(**metadata, records=records[offset:offset + limit],
                                        offset=offset, limit=limit)
    except (EvaluationDatasetError, FileNotFoundError) as error:
        raise _dataset_request_error(error) from error


def delete_evaluation_dataset(dataset_id: str) -> DeleteEvaluationDatasetResponse:
    try:
        delete_dataset(dataset_id)
    except (EvaluationDatasetError, FileNotFoundError) as error:
        raise _dataset_request_error(error) from error
    except OSError as error:
        logger.exception("Evaluation dataset deletion failed")
        raise HTTPException(status_code=503, detail="The evaluation dataset could not be deleted.") from error
    return DeleteEvaluationDatasetResponse(dataset_id=dataset_id, message="Evaluation dataset deleted.")


async def upload_evaluation_dataset(
    files: Annotated[list[UploadFile] | None, File()] = None,
    dataset_file: Annotated[UploadFile | None, File()] = None,
) -> UploadEvaluationDatasetResponse:
    """Accept a batch, retaining valid files and reporting errors per invalid file.

    The original single-file multipart field remains supported. Every accepted
    upload gets a new ID, even if another file has the same filename.
    """
    uploads = [*(files or []), *([dataset_file] if dataset_file else [])]
    accepted, errors = [], []
    try:
        if not uploads or len(uploads) > config.MAX_EVALUATION_UPLOAD_FILES:
            raise HTTPException(status_code=422, detail=(
                f"Select between 1 and {config.MAX_EVALUATION_UPLOAD_FILES} JSONL files per upload."
            ))
        for upload in uploads:
            filename = upload.filename or ""
            try:
                validate_upload_filename(filename)
                content = await upload.read(config.MAX_EVALUATION_DATASET_SIZE + 1)
                accepted.append(store_uploaded_dataset(filename, content))
            except EvaluationDatasetError as error:
                errors.append({"filename": filename, "error": str(error)})
        if not accepted:
            raise HTTPException(status_code=422, detail="; ".join(
                f"{item['filename']}: {item['error']}" for item in errors
            ))
        return UploadEvaluationDatasetResponse(datasets=accepted, errors=errors)
    finally:
        for upload in uploads:
            await upload.close()


def load_selected_datasets(
    request: RunEvaluationRequest,
    expected_hashes: dict[str, str] | None = None,
) -> list[tuple[dict, list]]:
    """Validate and snapshot all selections before any provider is invoked."""
    selected = []
    for dataset_id in request.dataset_ids:
        try:
            metadata = dataset_metadata(dataset_id)
            if metadata["validation_status"] != "valid":
                raise EvaluationDatasetError(f"{metadata['filename']}: {metadata['validation_error']}")
            content = dataset_path(dataset_id).read_bytes()
            digest = sha256(content).hexdigest()
            if digest != metadata["dataset_sha256"] or (
                expected_hashes is not None and expected_hashes.get(dataset_id) != digest
            ):
                raise EvaluationDatasetError(f"dataset {dataset_id} changed after selection; start a new run")
            records = parse_dataset_jsonl(content, maximum_records=config.MAX_EVALUATION_RECORDS)
        except (EvaluationDatasetError, FileNotFoundError) as error:
            raise _dataset_request_error(error) from error
        if request.record_ids is not None:
            requested_ids = set(request.record_ids)
            found_ids = {record.id for record in records}
            if missing := requested_ids - found_ids:
                raise HTTPException(status_code=404, detail=(
                    f"{metadata['filename']}: evaluation record IDs not found: {', '.join(sorted(missing))}"
                ))
            records = [record for record in records if record.id in requested_ids]
        selected.append((metadata, records))
    return selected


def execute_selected_evaluation(
    request: RunEvaluationRequest,
    *,
    expected_hashes: dict[str, str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> RunEvaluationResponse:
    selected = load_selected_datasets(request, expected_hashes)
    validate_evaluation_providers(request.answer_providers, request.judge_provider)
    total = sum(len(records) for _, records in selected) * len(request.answer_providers)
    completed = 0
    runs = []
    for metadata, records in selected:
        dataset_id = metadata["dataset_id"]

        def case_complete() -> None:
            nonlocal completed
            completed += 1
            if progress is not None:
                progress(completed, total, dataset_id)

        if progress is not None:
            progress(completed, total, dataset_id)
        try:
            run = run_evaluation(
                dataset_id, records, request.answer_providers, request.judge_provider,
                request.retrieval_configuration(), content_sha256=metadata["dataset_sha256"],
                on_case_complete=case_complete,
            )
        except Exception as error:
            logger.exception("On-demand evaluation failed")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="The evaluation could not be completed.") from error
        run.dataset_id = dataset_id
        run.dataset_name = metadata["filename"]
        for result in run.results:
            result.dataset_id = dataset_id
            result.dataset_name = metadata["filename"]
        runs.append(run)
    results = [result for run in runs for result in run.results]
    run_id = str(uuid4())
    aggregate = EvaluationRun(
        run_id=run_id, dataset_name=f"{len(runs)} selected datasets",
        dataset_sha256=sha256("\n".join(
            f"{run.dataset_id}:{run.dataset_sha256}" for run in runs
        ).encode()).hexdigest(),
        generated_at=datetime.now(timezone.utc), judge_provider=request.judge_provider,
        usage=summarize_llm_usage(usage for result in results
                                  for usage in (result.answer_usage, result.judge_usage)),
        summaries=[_summary(provider, results) for provider in request.answer_providers],
        results=results,
    )
    return RunEvaluationResponse(run_id=run_id, datasets=runs, aggregate=aggregate)


def run_evaluation_on_demand(request: RunEvaluationRequest) -> RunEvaluationResponse:
    return execute_selected_evaluation(request)
