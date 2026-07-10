"""JSONL serialization helpers for ingestion records."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TextIO, TypeVar

from pydantic import BaseModel

RecordT = TypeVar("RecordT", bound=BaseModel)


def dump_jsonl(records: list[BaseModel] | tuple[BaseModel, ...], output: TextIO) -> int:
    """Write Pydantic records to a text stream as JSON Lines."""

    count = 0
    for record in records:
        output.write(record.model_dump_json())
        output.write("\n")
        count += 1

    return count


def load_jsonl(input_stream: TextIO, model_type: type[RecordT]) -> list[RecordT]:
    """Read JSON Lines from a text stream into Pydantic records."""

    records: list[RecordT] = []
    for line_number, line in enumerate(input_stream, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        try:
            records.append(model_type.model_validate_json(stripped))
        except ValueError as exc:
            raise ValueError(f"Invalid JSONL record at line {line_number}") from exc

    return records


def records_to_jsonl(records: list[BaseModel] | tuple[BaseModel, ...]) -> str:
    """Serialize Pydantic records to a JSONL string."""

    buffer = StringIO()
    dump_jsonl(records, buffer)
    return buffer.getvalue()


def records_from_jsonl(payload: str, model_type: type[RecordT]) -> list[RecordT]:
    """Deserialize Pydantic records from a JSONL string."""

    return load_jsonl(StringIO(payload), model_type)


def write_jsonl(records: list[BaseModel] | tuple[BaseModel, ...], path: str | Path) -> int:
    """Write Pydantic records to a UTF-8 JSONL file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        return dump_jsonl(records, output)


def read_jsonl(path: str | Path, model_type: type[RecordT]) -> list[RecordT]:
    """Read Pydantic records from a UTF-8 JSONL file."""

    with Path(path).open(encoding="utf-8") as input_stream:
        return load_jsonl(input_stream, model_type)
