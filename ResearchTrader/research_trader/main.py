"""
ResearchTrader FastAPI Application
Main entry point for the API
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse

from research_trader.config import settings
from research_trader.router.qa import router as qa_router
from research_trader.router.search import router as search_router
from research_trader.router.strategy import router as strategy_router
from research_trader.router.summarize import router as summarize_router
from research_trader.utils.errors import ServiceError

# --- Logging Configuration --- >
log_level = settings.LOG_LEVEL.upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured with level: {log_level}")
# <---------------------------

# Create the FastAPI application
app = FastAPI(
    title="ResearchTrader API",
    description="API for discovering, exploring, and operationalizing ArXiv quantitative finance research",
    version="0.1.0",
    docs_url=None,
    redoc_url="/docs",
)

logger.info("Adding CORS middleware")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
logger.info("Including search router")
app.include_router(search_router, tags=["Search"])
app.include_router(summarize_router, tags=["Summarize"])
app.include_router(qa_router, tags=["QA"])
app.include_router(strategy_router, tags=["Strategy"])

# Add a simple root endpoint
@app.get("/")
def read_root():
    logger.debug("Root endpoint accessed")
    return {"message": "Welcome to ResearchTrader API"}

# Custom exception handler for ServiceError
@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "service": exc.service}
    )

# Custom Swagger UI with better styling
@app.get("/swagger", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.18.3/swagger-ui-bundle.js",
        swagger_css_url="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.18.3/swagger-ui.css",
    )

logger.info("FastAPI application initialized")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("research_trader.main:app", host="0.0.0.0", port=8000, reload=True)
