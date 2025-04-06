import sys
sys.path.append("/Users/arshath/play/insource/lenovo-scraper/v2.2")

import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from db import ProductLibraryDB

load_dotenv()

db_path = "/Users/arshath/play/insource/lenovo-scraper/nbs/v2.2/lenovo-scraper.db"
db = ProductLibraryDB(db_path=db_path)

MODEL = "openai:gpt-4o-mini"

class Laptop(BaseModel):
    vendor: str = Field(description="The vendor of the product", examples=["Lenovo", "Asus", "MSI"])
    country: str = Field(description="The country of the product", examples=["Australia", "USA", "UK"])
    price: float = Field(description="The price of the product, which should be after discount", examples=[1000.00])
    currency: str = Field(description="The currency of the product", examples=["AUD", "USD", "EUR"])
    mtm: Optional[str] = Field(description="The model of the product", examples=["IN LOQ 9th Ci5+2050"])
    series: Optional[str] = Field(description="The series of the product", examples=["LOQ 15IAX9"])
    size: Optional[str] = Field(description="The size of the product", examples=["15.6", "16.0", "17.3"])
    resolution: Optional[str] = Field(description="The resolution of the product", examples=["1920x1080", "2560x1440"])
    panel: Optional[str] = Field(description="The panel of the product", examples=["IPS", "TN", "VA"])
    touch_panel: Optional[bool] = Field(description="Whether the product has a touch panel", examples=[True, False])
    cpu: Optional[str] = Field(description="The CPU of the product", examples=["Intel Core i5-13650HX"])
    memory: Optional[str] = Field(description="The memory of the product", examples=["16GB", "32GB", "64GB"])
    gfx: Optional[str] = Field(description="The graphics card of the product", examples=["RTX 4060"])
    storage: str = Field(description="The storage of the product", examples=["1TB", "2TB", "4TB"])
    keyboard: Optional[str] = Field(description="The keyboard of the product", examples=["NBKB 1color backlight"])
    mouse: Optional[str] = Field(description="The mouse of the product", examples=["NBKB", "NBKB+Mouse"])
    wireless: Optional[str] = Field(description="The wireless of the product", examples=["2x2ax+BT"])
    lan: Optional[str] = Field(description="The LAN of the product", examples=["GbE", "10GbE"])
    os: Optional[str] = Field(description="The OS of the product", examples=["Windows 11", "Windows 10"])
    camera: Optional[str] = Field(description="The camera of the product", examples=["720p", "1080p"])
    office: Optional[str] = Field(description="The office of the product", examples=["Office 2021", "Office 2024"])
    warranty: Optional[str] = Field(description="The warranty of the product", examples=["1 year", "2 years"])
    battery: Optional[str] = Field(description="The battery of the product", examples=["5000mAh", "7000mAh"])
    weight: Optional[str] = Field(description="The weight of the product", examples=["1.5kg", "2.0kg"])
    color: Optional[str] = Field(description="The color of the product", examples=["Black", "White", "Red"])
    url: Optional[str] = Field(description="The URL of the product")

SYSTEM_PROMPT = """
    You are a helpful assistant that extracts structured information from e-commerce website data.
    
    <INSTRUCTIONS>
    - You are given information extracted from an e-commerce website.
    - Extract and normalize the information into a structured format.
    - If a field cannot be determined, use None or appropriate default values.
    - Normalize units and formats to be consistent.
    - Pay special attention to:
        * Extracting numeric values correctly
        * Converting prices to float values
        * Standardizing units (GB, TB, inches, etc.)
        * Identifying product specifications accurately
    - raw_data contains the following keys:
        * price
        * title
        * specs
        * alt_specs
        * json_ld
        * url
    - Some of the above raw_data might be None, so you should handle that.
    </INSTRUCTIONS>
    """

def prepare_user_prompt(raw_data: Dict[str, Any]) -> str:
    return f"""
    Here is the raw data:
    title: {raw_data.get("title")}
    specs: {raw_data.get("specs")}
    alt_specs: {raw_data.get("alt_specs")}
    json_ld: {raw_data.get("json_ld")}
    url: {raw_data.get("url")}
    """

agent = Agent(
    MODEL,
    result_type=Laptop,
    system_prompt=SYSTEM_PROMPT,
    )

raw_data = db.get_raw_data_by_run_id("20250403_083914")[-1]
user_prompt = prepare_user_prompt(raw_data)
result = asyncio.run(agent.run(user_prompt))
print(result)