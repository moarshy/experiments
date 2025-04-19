"""
Merged router for paper analysis and summarization with trading focus
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import os
from datetime import datetime
import logging
from typing import Optional, List

from research_trader.models.summary import PaperSummary, SummaryRequest, PaperText
from research_trader.services.arxiv_client import ArxivClient
from research_trader.services.pdf_service import PDFProcessor
from research_trader.services.openai_client import OpenAIClient
from research_trader.services.cache import CacheService
from research_trader.config import Settings, get_settings

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarize", tags=["summarize"])


@router.post("/", response_model=PaperSummary, summary="Generate trading-focused paper summary")
async def generate_paper_summary(
    request: SummaryRequest,
    config: Settings = Depends(get_settings)
):
    """
    Generate a comprehensive trading-focused summary of a paper.
    
    - **paper_id**: ArXiv paper ID
    - **force_refresh**: Force regeneration of summary even if cached
    
    Returns a detailed summary with:
    - Main objective
    - List of methods or approaches
    - List of key results
    - List of conclusions
    - List of potential trading applications
    - Implementation complexity assessment
    - Data requirements
    - Potential trading performance
    """
    # Check cache first (unless force refresh is requested)
    if not request.force_refresh:
        cached_summary = await CacheService.get_cached_summary(request.paper_id)
        if cached_summary:
            return cached_summary
    
    # Fetch paper if not already in cache
    cached_paper = await CacheService.get_cached_paper(request.paper_id)
    if cached_paper:
        paper = cached_paper
    else:
        arxiv_client = ArxivClient()
        paper = await arxiv_client.fetch_paper_by_id(request.paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper with ID {request.paper_id} not found")
            
        # Cache paper
        await CacheService.cache_paper(request.paper_id, paper)
    
    # Check if we have the full text
    paper_text = None
    has_full_text = False
    
    # Try to get the full text if available
    pdf_processor = PDFProcessor()
    try:
        paper_text = await pdf_processor.get_paper_text(paper.id, paper.pdf_url)
        has_full_text = True
        logger.info(f"Using full text for paper: {paper.id}")
    except Exception as e:
        logger.warning(f"Could not get full text for paper {paper.id}: {str(e)}")
        # Use abstract as fallback
        paper_text = paper.summary
    
    # Generate summary using OpenAI
    openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
    
    # Use the full paper text if available, otherwise use the abstract
    input_text = paper_text if has_full_text else paper.summary
    summary_data = await openai_client.generate_paper_summary_and_structure(
        {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "text": input_text,
            "is_full_text": has_full_text
        }
    )
    
    # Create response
    summary = PaperSummary(
        paper_id=paper.id,
        title=paper.title,
        objective=summary_data["objective"],
        methods=summary_data["methods"],
        results=summary_data["results"],
        conclusions=summary_data["conclusions"],
        trading_applications=summary_data["trading_applications"],
        summary=summary_data["summary"],
        keywords=summary_data["keywords"],
        implementation_complexity=summary_data["implementation_complexity"],
        data_requirements=summary_data["data_requirements"],
        potential_performance=summary_data["potential_performance"],
        has_full_text=has_full_text
    )
    
    # Cache summary
    await CacheService.cache_summary(request.paper_id, summary)
    
    return summary


@router.get("/{paper_id}", response_model=PaperSummary, summary="Get trading-focused paper summary")
async def get_paper_summary(
    paper_id: str,
    force_refresh: bool = False,
    config: Settings = Depends(get_settings)
):
    """
    Get a comprehensive trading-focused summary for a specific paper.
    
    - **paper_id**: ArXiv paper ID
    - **force_refresh**: Force regeneration of summary even if cached
    
    Returns a detailed trading-oriented summary if available, or generates one if not.
    """
    # Reuse the POST endpoint logic with a request object
    request = SummaryRequest(paper_id=paper_id, force_refresh=force_refresh)
    return await generate_paper_summary(request, config)


@router.post("/upload", response_model=PaperSummary, summary="Upload and summarize PDF")
async def upload_and_summarize(
    file: UploadFile = File(...),
    title: str = Form(...),
    authors: str = Form(...),
    config: Settings = Depends(get_settings)
):
    """
    Upload a research paper PDF and generate a trading-focused summary.
    
    - **file**: PDF file of the research paper
    - **title**: Title of the paper
    - **authors**: Paper authors (comma-separated)
    
    Returns a detailed trading-oriented summary of the uploaded paper.
    """
    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        try:
            # Read and save the file
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            
            # Process PDF
            pdf_processor = PDFProcessor()
            paper_text = await pdf_processor.extract_text_from_file(tmp_path)
            
            # Generate a paper ID for the uploaded paper
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            paper_id = f"uploaded_{timestamp}"
            
            # Split authors
            author_list = [a.strip() for a in authors.split(",")]
            
            # Generate summary
            openai_client = OpenAIClient(api_key=config.OPENAI_API_KEY, model=config.OPENAI_MODEL)
            summary_data = await openai_client.generate_paper_summary_and_structure(
                {
                    "id": paper_id,
                    "title": title,
                    "authors": author_list,
                    "text": paper_text,
                    "is_full_text": True
                }
            )
            
            # Create response
            summary = PaperSummary(
                paper_id=paper_id,
                title=title,
                objective=summary_data["objective"],
                methods=summary_data["methods"],
                results=summary_data["results"],
                conclusions=summary_data["conclusions"],
                trading_applications=summary_data["trading_applications"],
                summary=summary_data["summary"],
                keywords=summary_data["keywords"],
                implementation_complexity=summary_data["implementation_complexity"],
                data_requirements=summary_data["data_requirements"],
                potential_performance=summary_data["potential_performance"],
                has_full_text=True
            )
            
            # Store the paper text
            paper_text_obj = PaperText(
                paper_id=paper_id,
                title=title,
                abstract=summary_data["summary"],  # Use generated summary as abstract
                full_text=paper_text,
                extraction_date=datetime.now().isoformat()
            )
            
            # Cache the paper text and summary
            await CacheService.cache_paper_text(paper_id, paper_text_obj)
            await CacheService.cache_summary(paper_id, summary)
            
            return summary
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


@router.get("/text/{paper_id}", response_model=PaperText, summary="Get paper full text")
async def get_paper_text(
    paper_id: str,
    config: Settings = Depends(get_settings)
):
    """
    Get the full text of a specific paper.
    
    - **paper_id**: ArXiv paper ID or uploaded paper ID
    
    Returns the full text of the paper if available.
    """
    # Check if we have the paper text in cache
    cached_text = await CacheService.get_cached_paper_text(paper_id)
    if cached_text:
        return cached_text
    
    # Get paper metadata
    cached_paper = await CacheService.get_cached_paper(paper_id)
    if not cached_paper:
        arxiv_client = ArxivClient()
        paper = await arxiv_client.fetch_paper_by_id(paper_id)
        
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found")
            
        # Cache paper
        await CacheService.cache_paper(paper_id, paper)
        cached_paper = paper
    
    # Try to fetch and process the PDF
    pdf_processor = PDFProcessor()
    try:
        paper_text = await pdf_processor.get_paper_text(paper_id, cached_paper.pdf_url)
        
        # Create and cache the paper text object
        paper_text_obj = PaperText(
            paper_id=paper_id,
            title=cached_paper.title,
            abstract=cached_paper.summary,
            full_text=paper_text,
            extraction_date=datetime.now().isoformat()
        )
        
        await CacheService.cache_paper_text(paper_id, paper_text_obj)
        return paper_text_obj
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Could not retrieve full text for paper {paper_id}: {str(e)}"
        )