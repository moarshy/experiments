"""
Router for paper search endpoints
"""


from fastapi import APIRouter, Depends, Query, HTTPException
import logging

from research_trader.config import Settings, get_settings, settings
from research_trader.models.paper import Paper, PaperList
from research_trader.services.arxiv_client import ArxivClient
from research_trader.services.cache import CacheService

# Get logger for this module
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/", response_model=PaperList, summary="Search for papers")
async def search_papers(
    query: str = Query(..., min_length=1, description="Search query"),
    max_results: int = Query(
        settings.DEFAULT_PAPERS,
        ge=1,
        le=settings.MAX_PAPERS,
        description="Maximum number of results to return",
    ),
    config: Settings = Depends(get_settings),
):
    """
    Search for papers matching the query on ArXiv.

    - **query**: Search query string (required)
    - **max_results**: Maximum number of results to return (default: 10, max: 20)

    Returns a list of papers matching the query.
    """
    logger.info(f"Received search request. Query: '{query}', Max results: {max_results}")

    # Check cache first
    try:
        cached_results = await CacheService.get_cached_search(query, max_results)
        if cached_results:
            logger.info(f"Cache hit for query: '{query}', Max results: {max_results}")
            return cached_results
        logger.info(f"Cache miss for query: '{query}', Max results: {max_results}")
    except Exception as e:
        logger.warning(f"Cache lookup failed for query '{query}': {e}", exc_info=True)

    # Create ArXiv client and fetch papers
    try:
        arxiv_client = ArxivClient()
        papers = await arxiv_client.fetch_latest(query, max_results)
        logger.info(f"Fetched {len(papers)} papers from ArxivClient.")
    except Exception as e:
        logger.error(f"Failed to fetch papers from ArXiv for query '{query}': {e}", exc_info=True)
        # Re-raise as HTTPException or return an error response
        raise HTTPException(status_code=503, detail="Failed to fetch papers from ArXiv.")

    # Create response
    response = PaperList(papers=papers, total_count=len(papers), query=query)

    # Cache results
    try:
        await CacheService.cache_search(query, max_results, response)
        logger.info(f"Cached results for query: '{query}', Max results: {max_results}")
    except Exception as e:
        logger.warning(f"Failed to cache results for query '{query}': {e}", exc_info=True)

    logger.info(f"Returning {len(papers)} papers for query: '{query}'")
    return response


@router.get("/{paper_id}", response_model=Paper, summary="Get paper by ID")
async def get_paper_by_id(paper_id: str, config: Settings = Depends(get_settings)):
    """
    Get a specific paper by its ArXiv ID.

    - **paper_id**: ArXiv paper ID (can be full URL or just the ID part)

    Returns the paper details if found.
    """
    logger.info(f"Received request for paper ID: {paper_id}")

    # Check cache first
    try:
        cached_paper = await CacheService.get_cached_paper(paper_id)
        if cached_paper:
            logger.info(f"Cache hit for paper ID: {paper_id}")
            return cached_paper
        logger.info(f"Cache miss for paper ID: {paper_id}")
    except Exception as e:
        logger.warning(f"Cache lookup failed for paper ID '{paper_id}': {e}", exc_info=True)

    # Create ArXiv client and fetch paper
    try:
        arxiv_client = ArxivClient()
        paper = await arxiv_client.fetch_paper_by_id(paper_id)
        if not paper:
            logger.warning(f"Paper ID not found via ArXiv: {paper_id}")
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found.")
        logger.debug(f"Fetched paper {paper_id} from ArxivClient.")
    except HTTPException: # Re-raise HTTPException
        raise
    except Exception as e:
        logger.error(f"Failed to fetch paper ID '{paper_id}' from ArXiv: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Failed to fetch paper ID {paper_id} from ArXiv.")

    # Cache paper
    try:
        await CacheService.cache_paper(paper_id, paper)
        logger.info(f"Cached paper ID: {paper_id}")
    except Exception as e:
        logger.warning(f"Failed to cache paper ID '{paper_id}': {e}", exc_info=True)

    logger.info(f"Returning details for paper ID: {paper_id}")
    return paper
