"""
Enhanced caching service for ResearchTrader API responses
"""
import json
import time
import hashlib
from typing import Any, Dict, Optional, TypeVar, Generic, Union
from asyncio import Lock

from research_trader.config import settings
from research_trader.models.summary import PaperText

T = TypeVar('T')


class MemoryCache:
    """Simple in-memory cache implementation"""
    
    def __init__(self, ttl: int = settings.CACHE_TTL):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        self.lock = Lock()
    
    def _generate_key(self, key_parts: Union[str, Dict[str, Any]]) -> str:
        """Generate a cache key from string or dictionary"""
        if isinstance(key_parts, dict):
            # Sort keys for consistent hashing
            serialized = json.dumps(key_parts, sort_keys=True)
        else:
            serialized = str(key_parts)
            
        return hashlib.md5(serialized.encode()).hexdigest()
    
    async def get(self, key_parts: Union[str, Dict[str, Any]]) -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key_parts: String or dictionary used to generate the cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        key = self._generate_key(key_parts)
        
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if entry is expired
                if time.time() < entry["expiry"]:
                    return entry["value"]
                else:
                    # Remove expired entry
                    del self.cache[key]
                    
        return None
    
    async def set(
        self, 
        key_parts: Union[str, Dict[str, Any]], 
        value: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """
        Set a value in cache
        
        Args:
            key_parts: String or dictionary used to generate the cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        key = self._generate_key(key_parts)
        expiry = time.time() + (ttl if ttl is not None else self.ttl)
        
        async with self.lock:
            self.cache[key] = {
                "value": value,
                "expiry": expiry
            }
    
    async def delete(self, key_parts: Union[str, Dict[str, Any]]) -> bool:
        """
        Delete a value from cache
        
        Args:
            key_parts: String or dictionary used to generate the cache key
            
        Returns:
            True if entry was found and deleted, False otherwise
        """
        key = self._generate_key(key_parts)
        
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
                
        return False
    
    async def clear(self) -> None:
        """Clear all cached values"""
        async with self.lock:
            self.cache.clear()


# Create a singleton cache instance
memory_cache = MemoryCache()


class CacheService:
    """Service for caching API responses"""
    
    @staticmethod
    async def get_cached_search(query: str, max_results: int) -> Optional[Any]:
        """Get cached search results"""
        if not settings.ENABLE_CACHE:
            return None
            
        key = {
            "type": "search",
            "query": query,
            "max_results": max_results
        }
        return await memory_cache.get(key)
    
    @staticmethod
    async def cache_search(query: str, max_results: int, results: Any) -> None:
        """Cache search results"""
        if not settings.ENABLE_CACHE:
            return
            
        key = {
            "type": "search",
            "query": query,
            "max_results": max_results
        }
        await memory_cache.set(key, results)
    
    @staticmethod
    async def get_cached_paper(paper_id: str) -> Optional[Any]:
        """Get cached paper details"""
        if not settings.ENABLE_CACHE:
            return None
            
        key = {
            "type": "paper",
            "id": paper_id
        }
        return await memory_cache.get(key)
    
    @staticmethod
    async def cache_paper(paper_id: str, paper_data: Any) -> None:
        """Cache paper details"""
        if not settings.ENABLE_CACHE:
            return
            
        key = {
            "type": "paper",
            "id": paper_id
        }
        await memory_cache.set(key, paper_data)
    
    @staticmethod
    async def get_cached_structure(paper_id: str) -> Optional[Any]:
        """Get cached paper structure"""
        if not settings.ENABLE_CACHE:
            return None
            
        key = {
            "type": "structure",
            "paper_id": paper_id
        }
        return await memory_cache.get(key)
    
    @staticmethod
    async def cache_structure(paper_id: str, structure: Any) -> None:
        """Cache paper structure"""
        if not settings.ENABLE_CACHE:
            return
            
        key = {
            "type": "structure",
            "paper_id": paper_id
        }
        await memory_cache.set(key, structure)
    
    @staticmethod
    async def get_cached_summary(paper_id: str) -> Optional[Any]:
        """Get cached paper summary"""
        if not settings.ENABLE_CACHE:
            return None
            
        key = {
            "type": "summary",
            "paper_id": paper_id
        }
        return await memory_cache.get(key)
    
    @staticmethod
    async def cache_summary(paper_id: str, summary: Any) -> None:
        """Cache paper summary"""
        if not settings.ENABLE_CACHE:
            return
            
        key = {
            "type": "summary",
            "paper_id": paper_id
        }
        await memory_cache.set(key, summary)
        
    @staticmethod
    async def get_cached_paper_text(paper_id: str) -> Optional[PaperText]:
        """Get cached paper full text"""
        if not settings.ENABLE_CACHE:
            return None
            
        key = {
            "type": "paper_text",
            "paper_id": paper_id
        }
        return await memory_cache.get(key)
    
    @staticmethod
    async def cache_paper_text(paper_id: str, paper_text: PaperText) -> None:
        """Cache paper full text"""
        if not settings.ENABLE_CACHE:
            return
            
        key = {
            "type": "paper_text",
            "paper_id": paper_id
        }
        # Use a longer TTL for paper text since it's expensive to reprocess
        paper_text_ttl = settings.CACHE_TTL * 2
        await memory_cache.set(key, paper_text, ttl=paper_text_ttl)