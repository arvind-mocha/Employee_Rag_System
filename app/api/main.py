"""FastAPI application entry point."""

import uvicorn
from fastapi import FastAPI

from app.api.router import api_router


def create_app() -> FastAPI:
    """Create and configure the API application."""
    application = FastAPI(title="Employee RAG Data API")
    application.include_router(api_router)
    return application


app = create_app()


def main() -> None:
    """Run the API using the project's command-line script."""
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
