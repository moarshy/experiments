"""
Router for paper Q&A endpoints
"""


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from research_trader.config import Settings, get_settings
from research_trader.models.summary import QAResponse
from research_trader.services.arxiv_client import ArxivClient
from research_trader.services.cache import CacheService
from research_trader.services.openai_client import OpenAIClient


class QARequest(BaseModel):
    """Request model for Q&A"""

    question: str
    paper_ids: list[str]


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/", response_model=QAResponse, summary="Ask a question about papers")
async def ask_question(request: QARequest, config: Settings = Depends(get_settings)):
    """
    Ask a question about the provided papers.

    - **question**: User's question
    - **paper_ids**: List of ArXiv paper IDs to consider

    Returns an answer based on the content of the specified papers.
    """
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper ID is required")

    # Collect paper information for context
    paper_contexts = []

    for paper_id in request.paper_ids:
        # Try to get the summary first (it includes structured sections)
        cached_summary = await CacheService.get_cached_summary(paper_id)
        if cached_summary:
            paper_contexts.append(cached_summary)
            continue

        # If no summary, try to get the paper and structure
        cached_paper = await CacheService.get_cached_paper(paper_id)
        if not cached_paper:
            # Fetch paper from ArXiv
            arxiv_client = ArxivClient(base_url=config.ARXIV_API_URL)
            paper = await arxiv_client.fetch_paper_by_id(paper_id)

            if not paper:
                raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")

            await CacheService.cache_paper(paper_id, paper)
            cached_paper = paper

        # Get structure if available
        cached_structure = await CacheService.get_cached_structure(paper_id)
        if not cached_structure:
            # Extract structure using OpenAI
            openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
            structure_dict = await openai_client.extract_paper_structure(cached_paper.summary)

            # Cache structure
            await CacheService.cache_structure(paper_id, structure_dict)
            cached_structure = structure_dict

        # Add to context
        paper_context = {
            "id": cached_paper.id,
            "title": cached_paper.title,
            "authors": cached_paper.authors,
            "summary": cached_paper.summary,
            "sections": cached_structure,
        }
        paper_contexts.append(paper_context)

    # Generate answer using OpenAI
    openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
    answer_data = await openai_client.answer_question(request.question, paper_contexts)

    # Create response
    response = QAResponse(
        question=request.question,
        answer=answer_data["answer"],
        sources=[
            request.paper_ids[i - 1]
            for i in answer_data["sources"]
            if 0 < i <= len(request.paper_ids)
        ],
        confidence=answer_data["confidence"],
    )

    return response
