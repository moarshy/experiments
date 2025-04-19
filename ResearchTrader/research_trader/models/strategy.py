"""
Strategy generation models for ResearchTrader
"""


from pydantic import BaseModel, Field


class StrategyRequest(BaseModel):
    """Request model for strategy generation"""

    paper_ids: list[str] = Field(..., description="IDs of papers to base strategy on")
    market: str = Field(default="equities", description="Target market (equities, forex, crypto)")
    timeframe: str = Field(
        default="daily", description="Trading timeframe (tick, minute, hourly, daily)"
    )
    risk_profile: str = Field(
        default="moderate", description="Risk profile (conservative, moderate, aggressive)"
    )
    additional_context: str | None = Field(
        None, description="Additional user context or requirements"
    )

    class Config:
        schema_extra = {
            "example": {
                "paper_ids": ["http://arxiv.org/abs/2204.11824", "http://arxiv.org/abs/2105.12780"],
                "market": "crypto",
                "timeframe": "hourly",
                "risk_profile": "aggressive",
                "additional_context": "Focus on momentum strategies with strict stop-loss mechanisms",
            }
        }


class StrategyResponse(BaseModel):
    """Response model for generated trading strategy"""

    strategy_name: str
    description: str
    python_code: str
    paper_references: list[str]
    usage_notes: str
    limitations: str

    class Config:
        schema_extra = {
            "example": {
                "strategy_name": "Adaptive Momentum with RL Position Sizing",
                "description": "A trading strategy combining adaptive momentum indicators with reinforcement learning for position sizing, based on techniques from papers X and Y.",
                "python_code": "import pandas as pd\nimport numpy as np\n\nclass AdaptiveMomentumStrategy:\n    def __init__(self, lookback=20):\n        self.lookback = lookback\n    \n    def generate_signals(self, data):\n        # Strategy implementation\n        return signals",
                "paper_references": [
                    "http://arxiv.org/abs/2204.11824",
                    "http://arxiv.org/abs/2105.12780",
                ],
                "usage_notes": "This strategy requires minute-level OHLCV data and should be recalibrated monthly.",
                "limitations": "Performance degrades in low-volatility regimes. Not suitable for illiquid assets.",
            }
        }
