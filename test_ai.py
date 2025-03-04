from pathlib import Path
from typing import Optional
from loguru import logger
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request, HTTPException
from fastapi.staticfiles import StaticFiles
import os

from genflow.api import router
from genflow.api.v1 import flow_app_openapi_router

from genflow.interface.utils import setup_llm_caching
from genflow.services.utils import initialize_services
from genflow.services.plugins.langfuse import LangfuseInstance
from genflow.services.utils import teardown_services
from genflow.utils.logger import configure


def create_app():
    """Create the FastAPI app and include the router."""

    configure()

    app = FastAPI()

    origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    app.include_router(flow_app_openapi_router)

    app.on_event("startup")(initialize_services)
    app.on_event("startup")(setup_llm_caching)
    app.on_event("startup")(LangfuseInstance.update)

    app.on_event("shutdown")(teardown_services)
    app.on_event("shutdown")(LangfuseInstance.teardown)

    @app.exception_handler(HTTPException)
    async def custom_exception_handler(request: Request, exc: HTTPException):
        # Return a generic message for all 500 errors
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            # Log the original error before sending the generic message
            logger.error(exc.detail)
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": "Something went wrong. Please try again. If this issue persists please contact us at support@aiplanet.com ."
                },
            )

        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return app


def setup_static_files(app: FastAPI, static_files_dir: Path):
    """
    Setup the static files directory.
    Args:
        app (FastAPI): FastAPI app.
        path (str): Path to the static files directory.
    """
    app.mount(
        "/",
        StaticFiles(directory=static_files_dir, html=True),
        name="static",
    )

    @app.exception_handler(404)
    async def custom_404_handler(request, exc: HTTPException):
        path = static_files_dir / "index.html"

        if request.url.path.startswith("/api"):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        else:
            if not path.exists():
                raise RuntimeError(f"File at path {path} does not exist.")
            return FileResponse(path)


def get_static_files_dir():
    """Get the static files directory relative to genflow's main.py file."""
    frontend_path = Path(__file__).parent
    return frontend_path / "frontend"


def setup_app(static_files_dir: Optional[Path] = None, backend_only: bool = False) -> FastAPI:
    """Setup the FastAPI app."""
    # get the directory of the current file
    if not static_files_dir:
        static_files_dir = get_static_files_dir()

    if not backend_only and (not static_files_dir or not static_files_dir.exists()):
        raise RuntimeError(f"Static files directory {static_files_dir} does not exist.")
    app = create_app()

    if not backend_only and static_files_dir is not None:
        setup_static_files(app, static_files_dir)
    return app


def create_production_app() -> FastAPI:
    """Setup the FastAPI app."""
    # get the directory of the current file
    from dotenv import load_dotenv

    load_dotenv("./.env")
    configure(log_file=os.environ.get("genflow_LOG_FILE"))

    static_files_dir = Path(os.environ.get("genflow_FRONTEND_PATH"))
    if not static_files_dir:
        static_files_dir = get_static_files_dir()

    if not static_files_dir.exists():
        raise RuntimeError(f"Static files directory {static_files_dir} does not exist.")
    app = create_app()
    setup_static_files(app, static_files_dir)
    return app


app = create_app()  # Top-level `app` instance for uvicorn

if __name__ == "__main__":
    import uvicorn
    from genflow.__main__ import get_number_of_workers

    configure()
    uvicorn.run(
        app,  # Use the app object here
        host="127.0.0.1",
        port=7860,
        log_level="debug",
        reload=True,
    )
