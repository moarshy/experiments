"""
Router for trading strategy generation endpoints
"""


from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from research_trader.config import Settings, get_settings
from research_trader.models.strategy import StrategyRequest, StrategyResponse
from research_trader.services.arxiv_client import ArxivClient
from research_trader.services.cache import CacheService
from research_trader.services.openai_client import OpenAIClient

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.post("/", response_model=StrategyResponse, summary="Generate trading strategy")
async def generate_strategy(request: StrategyRequest, config: Settings = Depends(get_settings)):
    """
    Generate a trading strategy based on the specified papers.

    - **paper_ids**: List of ArXiv paper IDs to base the strategy on
    - **market**: Target market (equities, forex, crypto)
    - **timeframe**: Trading timeframe (tick, minute, hourly, daily)
    - **risk_profile**: Risk profile (conservative, moderate, aggressive)
    - **additional_context**: Additional user context or requirements

    Returns a complete trading strategy with Python code, usage notes, and limitations.
    """
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper ID is required")

    # Collect paper summaries for context
    paper_summaries = []

    for paper_id in request.paper_ids:
        # Try to get the summary
        cached_summary = await CacheService.get_cached_summary(paper_id)
        if cached_summary:
            paper_summaries.append(cached_summary)
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

        # Add minimal summary to context
        paper_summary = {
            "id": cached_paper.id,
            "title": cached_paper.title,
            "summary": cached_paper.summary,
            "sections": cached_structure,
        }
        paper_summaries.append(paper_summary)

    # Generate strategy
    openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)

    # Convert request to dict format expected by OpenAI client
    request_dict = {
        "market": request.market,
        "timeframe": request.timeframe,
        "risk_profile": request.risk_profile,
        "additional_context": request.additional_context,
    }

    # Full response will be built during streaming
    full_response = ""
    strategy_generator = openai_client.generate_trading_strategy(paper_summaries, request_dict)

    async for chunk in strategy_generator:
        full_response += chunk

    # Parse the response to extract components
    # We'll try to be smart about identifying parts from the unstructured text
    python_code_start = full_response.find("```python")
    if python_code_start != -1:
        python_code_end = full_response.find("```", python_code_start + 8)
        python_code = full_response[python_code_start + 8 : python_code_end].strip()
    else:
        # Fallback: look for any code block
        python_code_start = full_response.find("```")
        if python_code_start != -1:
            python_code_end = full_response.find("```", python_code_start + 3)
            python_code = full_response[python_code_start + 3 : python_code_end].strip()
        else:
            # No code block found, try to extract based on imports
            lines = full_response.split("\n")
            code_start = -1
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    code_start = i
                    break

            if code_start != -1:
                python_code = "\n".join(lines[code_start:])
            else:
                python_code = (
                    "# No code could be extracted automatically\n# Please review the full response"
                )

    # Extract strategy name from the beginning
    lines = full_response.split("\n")
    strategy_name = "Trading Strategy"
    for line in lines[:10]:  # Look in first 10 lines
        if line.strip() and not line.startswith("#") and not line.startswith("```"):
            strategy_name = line.strip()
            break

    # Extract description - text before code block
    if python_code_start != -1:
        description = full_response[:python_code_start].strip()
    else:
        # Take first few paragraphs as description
        paragraphs = [p for p in full_response.split("\n\n") if p.strip()]
        description = "\n\n".join(paragraphs[:3]) if paragraphs else "No description available"

    # Look for usage notes and limitations near the end
    usage_notes = "See full response for usage details"
    limitations = "See full response for limitations"

    lower_response = full_response.lower()

    # Check for usage notes section
    usage_notes_keywords = ["usage", "how to use", "parameters", "implementation"]
    for keyword in usage_notes_keywords:
        pos = lower_response.find(keyword)
        if pos != -1:
            end_pos = lower_response.find("#", pos)
            if end_pos == -1:
                end_pos = len(lower_response)

            # Extract paragraph
            paragraph_start = lower_response.rfind("\n\n", 0, pos)
            if paragraph_start == -1:
                paragraph_start = 0

            paragraph_end = lower_response.find("\n\n", pos)
            if paragraph_end == -1:
                paragraph_end = len(lower_response)

            usage_notes = full_response[paragraph_start:paragraph_end].strip()
            break

    # Check for limitations section
    limitations_keywords = ["limitation", "caveat", "consideration", "warning"]
    for keyword in limitations_keywords:
        pos = lower_response.find(keyword)
        if pos != -1:
            # Extract paragraph
            paragraph_start = lower_response.rfind("\n\n", 0, pos)
            if paragraph_start == -1:
                paragraph_start = 0

            paragraph_end = lower_response.find("\n\n", pos)
            if paragraph_end == -1:
                paragraph_end = len(lower_response)

            limitations = full_response[paragraph_start:paragraph_end].strip()
            break

    # Create response
    response = StrategyResponse(
        strategy_name=strategy_name,
        description=description,
        python_code=python_code,
        paper_references=request.paper_ids,
        usage_notes=usage_notes,
        limitations=limitations,
    )

    return response


@router.post("/stream", summary="Generate trading strategy (streaming)")
async def generate_strategy_stream(
    request: StrategyRequest, config: Settings = Depends(get_settings)
):
    """
    Generate a trading strategy with streaming response.

    Same parameters as the non-streaming endpoint, but returns chunks of the
    generated strategy as they are produced.
    """
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="At least one paper ID is required")

    # Collect paper summaries for context (same as non-streaming endpoint)
    paper_summaries = []

    for paper_id in request.paper_ids:
        # Try to get the summary
        cached_summary = await CacheService.get_cached_summary(paper_id)
        if cached_summary:
            paper_summaries.append(cached_summary)
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

        # Add minimal summary to context
        paper_summary = {
            "id": cached_paper.id,
            "title": cached_paper.title,
            "summary": cached_paper.summary,
            "sections": cached_structure,
        }
        paper_summaries.append(paper_summary)

    # Generate strategy with streaming
    openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)

    # Convert request to dict format expected by OpenAI client
    request_dict = {
        "market": request.market,
        "timeframe": request.timeframe,
        "risk_profile": request.risk_profile,
        "additional_context": request.additional_context,
    }

    async def stream_generator():
        strategy_generator = openai_client.generate_trading_strategy(paper_summaries, request_dict)
        async for chunk in strategy_generator:
            yield chunk

    return StreamingResponse(stream_generator(), media_type="text/plain")
