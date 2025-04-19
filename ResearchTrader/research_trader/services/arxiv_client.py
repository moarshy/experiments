"""
ArXiv API client using the arxiv package
"""
import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

import arxiv
from arxiv.arxiv import Result as ArxivResult

from research_trader.models.paper import Paper
from research_trader.config import settings
from research_trader.utils.errors import ServiceError


class ArxivClient:
    """Client for interacting with the ArXiv API using the arxiv package"""
    
    def __init__(self):
        self.last_request_time = 0
        self.min_request_interval = 3  # Seconds between requests to respect rate limits
    
    async def _enforce_rate_limit(self):
        """Ensure we don't exceed ArXiv's rate limits"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
    
    def _convert_to_paper(self, arxiv_result: ArxivResult) -> Paper:
        """Convert arxiv.Result to our Paper model"""
        # Extract the PDF URL if available
        pdf_url = arxiv_result.pdf_url if hasattr(arxiv_result, 'pdf_url') else None
        
        # Extract primary category
        category = "q-fin"
        if hasattr(arxiv_result, 'primary_category'):
            category = arxiv_result.primary_category
        
        return Paper(
            id=str(arxiv_result.entry_id),
            title=arxiv_result.title,
            authors=[author.name for author in arxiv_result.authors],
            summary=arxiv_result.summary,
            published=arxiv_result.published,
            link=arxiv_result.entry_id,
            category=category,
            pdf_url=pdf_url
        )
    
    async def fetch_latest(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Fetch the latest papers from ArXiv matching the query
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of Paper objects
            
        Raises:
            ServiceError: If ArXiv API request fails
        """
        await self._enforce_rate_limit()
        
        # Format the search query for arxiv package
        formatted_query = f"cat:q-fin* AND {query}"
        
        # Run the search in a separate thread to avoid blocking the event loop
        try:
            # Use arxiv.Search with appropriate parameters
            search = arxiv.Search(
                query=formatted_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # Execute search and convert results to our Paper model
            # We need to run this in an executor since arxiv.Search is synchronous
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: list(search.results()))
            
            papers = [self._convert_to_paper(result) for result in results]
            return papers
            
        except Exception as e:
            error_msg = f"Error fetching papers from ArXiv: {str(e)}"
            raise ServiceError(
                detail=error_msg,
                service="arxiv",
                status_code=503
            )
    
    async def fetch_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """
        Fetch a specific paper by its ID
        
        Args:
            paper_id: ArXiv paper ID
            
        Returns:
            Paper object or None if not found
        """
        await self._enforce_rate_limit()
        
        # Extract the bare ID without the domain prefix
        bare_id = paper_id.split("/")[-1]
        if "abs" in bare_id:
            bare_id = bare_id.split("abs/")[-1]
        
        try:
            # Use arxiv.Search with paper ID
            search = arxiv.Search(id_list=[bare_id])
            
            # Execute search and convert results to our Paper model
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: list(search.results()))
            
            if not results:
                return None
                
            return self._convert_to_paper(results[0])
            
        except Exception as e:
            error_msg = f"Error fetching paper from ArXiv: {str(e)}"
            raise ServiceError(
                detail=error_msg,
                service="arxiv",
                status_code=503
            )