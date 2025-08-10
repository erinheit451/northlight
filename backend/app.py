# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import APP_TITLE, APP_VERSION, ALLOW_ORIGINS
from backend.routers import diagnose, export
from backend.data.loader import init_data, data_meta, get_bench

# Anchors cache is optional (Phase 2). Import defensively so Phase 1 still runs.
try:
    from backend.data.anchors import init_anchor_cache  # type: ignore
except Exception:  # module may not exist yet
    init_anchor_cache = None  # noqa: N816


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, version=APP_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        # Load benchmark data
        init_data()
        # Build anchors cache if available (Phase 2+)
        if init_anchor_cache is not None:
            try:
                init_anchor_cache(get_bench())
            except Exception:
                # Optional: add logging here if you want visibility
                # e.g., import logging; logging.getLogger(__name__).exception("Anchor cache init failed")
                pass

    @app.get("/meta")
    def meta():
        """Lightweight health/meta endpoint."""
        return data_meta()

    # API routes
    app.include_router(diagnose.router)
    app.include_router(export.router)

    return app


app = create_app()
