"""Declarative API route definition."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ApiRoute:
    """Metadata needed to register one API handler."""

    name: str
    path: str
    endpoint: Callable[..., Any]
    response_model: type[Any] | None = None
    status_code: int | None = None
    tags: tuple[str, ...] = ()
