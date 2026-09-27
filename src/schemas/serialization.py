"""Serialization and deserialization helpers for domain schemas."""

import json
from typing import Any, Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def to_json(model: BaseModel, indent: int | None = None) -> str:
    """Serialize a domain model instance to a valid JSON string using Pydantic."""
    return model.model_dump_json(indent=indent)


def to_dict(model: BaseModel) -> dict[str, Any]:
    """Serialize a domain model instance to a Python dictionary."""
    return model.model_dump(mode="python")


def from_json(model_cls: Type[T], json_str: str) -> T:
    """Deserialize a JSON string into a validated domain model instance.

    Raises:
        ValueError: If JSON is malformed or violates schema validation constraints.
    """
    if not isinstance(json_str, str) or not json_str.strip():
        raise ValueError("Cannot parse empty or non-string input into schema.")

    try:
        return model_cls.model_validate_json(json_str)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to validate JSON for {model_cls.__name__}: {exc}") from exc


def from_dict(model_cls: Type[T], data: dict[str, Any]) -> T:
    """Deserialize a Python dictionary into a validated domain model instance.

    Raises:
        ValueError: If dictionary violates schema validation constraints.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict input for {model_cls.__name__}, got {type(data).__name__}")

    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Failed to validate dict for {model_cls.__name__}: {exc}") from exc
