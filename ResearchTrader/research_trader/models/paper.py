"""
Paper data models for ResearchTrader
"""

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Paper(BaseModel):
    """Representation of an ArXiv research paper"""

    id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime
    link: HttpUrl
    category: str = Field(default="q-fin")
    pdf_url: HttpUrl = Field(default=None)

    @field_validator("published")
    def parse_datetime(cls, value):
        """Parse datetime from ArXiv string format if needed"""
        if isinstance(value, datetime):
            return value

        # Handle ISO format
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            pass

        # Try other common formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format without timezone
            "%Y-%m-%d %H:%M:%S",  # Common format
            "%a, %d %b %Y %H:%M:%S %Z",  # ArXiv format
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except (ValueError, TypeError):
                continue

        raise ValueError(f"Unable to parse datetime from {value}")

    class Config:
        schema_extra = {
            "example": {
                "id": "http://arxiv.org/abs/2204.11824",
                "title": "Deep Reinforcement Learning for Algorithmic Trading",
                "authors": ["John Smith", "Jane Doe"],
                "summary": "This paper explores deep reinforcement learning...",
                "published": "2023-08-15T12:30:45Z",
                "link": "https://arxiv.org/abs/2204.11824",
                "category": "q-fin.TR",
                "pdf_url": "https://arxiv.org/pdf/2204.11824.pdf",
            }
        }


class PaperList(BaseModel):
    """Response model for list of papers"""

    papers: list[Paper]
    total_count: int
    query: str
