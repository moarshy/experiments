"""
Merged summary models for paper information with trading focus
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PaperSummary(BaseModel):
    """Comprehensive summary of a research paper for trading applications"""
    paper_id: str = Field(..., description="The ID of the original paper")
    title: str = Field(..., description="The title of the paper")
    
    # Core information
    objective: str = Field(..., description="The main objective or purpose of the paper")
    methods: List[str] = Field(..., description="List of methods or approaches used in the paper")
    results: List[str] = Field(..., description="List of key results or findings of the paper")
    conclusions: List[str] = Field(..., description="List of conclusions or implications of the paper")
    trading_applications: List[str] = Field(..., description="List of potential trading applications of the paper's findings")
    
    # Additional analysis
    summary: str = Field(..., description="A concise human-readable summary of the paper")
    keywords: List[str] = Field(default_factory=list, description="Key themes and topics")
    implementation_complexity: str = Field(..., description="Assessment of implementation difficulty (Low/Medium/High)")
    data_requirements: List[str] = Field(..., description="List of data needed to implement the approach")
    potential_performance: str = Field(..., description="Expected trading performance based on paper results")
    
    # Paper content tracking
    has_full_text: bool = Field(default=False, description="Whether the full text is available")
    
    class Config:
        schema_extra = {
            "example": {
                "paper_id": "http://arxiv.org/abs/2204.11824",
                "title": "Deep Reinforcement Learning for Algorithmic Trading",
                "objective": "This paper aims to develop a novel reinforcement learning framework for algorithmic trading.",
                "methods": [
                    "Deep Q-Network architecture with temporal attention mechanism",
                    "Custom reward function based on Sharpe ratio and drawdown",
                    "Multi-asset portfolio optimization"
                ],
                "results": [
                    "15% improvement in Sharpe ratio compared to baseline models",
                    "Reduced maximum drawdown by 8% during market stress periods",
                    "Improved sample efficiency by 35% through replay buffer prioritization"
                ],
                "conclusions": [
                    "RL-based trading strategies outperform traditional methods in volatile markets",
                    "Attention mechanisms help identify relevant price patterns across timeframes",
                    "The approach is generalizable across multiple asset classes"
                ],
                "trading_applications": [
                    "Intraday trading strategies for liquid markets",
                    "Cross-asset portfolio allocation optimization",
                    "Dynamic risk management during market regime changes"
                ],
                "summary": "This research introduces a deep reinforcement learning approach for algorithmic trading that leverages temporal attention mechanisms to process market data. Testing across equity, forex, and cryptocurrency markets shows consistent outperformance versus traditional methods, with particular strength in high-volatility conditions.",
                "keywords": ["reinforcement learning", "algorithmic trading", "DQN", "attention mechanism", "multi-asset"],
                "implementation_complexity": "Medium",
                "data_requirements": ["Minute-level OHLCV data", "Order book depth (L2 data)", "Market sentiment indicators"],
                "potential_performance": "High potential for liquid markets with clear regime changes, moderate for stable markets",
                "has_full_text": True
            }
        }


class SummaryRequest(BaseModel):
    """Request model for paper summarization"""
    paper_id: str
    force_refresh: bool = Field(default=False, description="Force refresh the summary even if cached")


class PaperText(BaseModel):
    """Model for paper full text storage"""
    paper_id: str = Field(..., description="The ID of the paper")
    title: str = Field(..., description="The title of the paper")
    abstract: str = Field(..., description="The abstract of the paper")
    full_text: str = Field(..., description="The full text of the paper")
    sections: Dict[str, str] = Field(default_factory=dict, description="Paper sections if available")
    extraction_date: str = Field(..., description="When the text was extracted")
    
    class Config:
        schema_extra = {
            "example": {
                "paper_id": "http://arxiv.org/abs/2204.11824",
                "title": "Deep Reinforcement Learning for Algorithmic Trading",
                "abstract": "In this paper, we present a deep reinforcement learning approach for algorithmic trading...",
                "full_text": "1. Introduction\nAlgorithmic trading has become increasingly important...",
                "sections": {
                    "introduction": "Algorithmic trading has become increasingly important...",
                    "methodology": "Our approach uses a deep Q-network architecture...",
                    "results": "We evaluated our approach on historical market data..."
                },
                "extraction_date": "2025-04-19T12:34:56"
            }
        }


class QAResponse(BaseModel):
    """Response model for Q&A queries"""
    question: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    suggestions: List[str] = Field(default_factory=list, description="Follow-up questions or areas to explore")