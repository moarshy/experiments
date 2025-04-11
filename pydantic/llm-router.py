import asyncio
from enum import Enum
import os
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

SINGLE_QUERY_SYSTEM_PROMPT = """
You are an AI assistant that converts user queries into structured search filters. Your task is to extract and format the following information from each query:

1. **Document Type**: Identify a document type that exactly matches one of the available types listed below.
2. **Document Date**: Extract any specific date or date range in the format DD/MM/YYYY.
3. **Total Amount**: Identify any amount or amount range (numeric value only) with qualifiers (e.g., "greater than", "less than", "at least", "at most").
4. **Filterability**: Set `can_be_filtered` to True if the query includes valid filter criteria (document type, date, or amount). Otherwise, set it to False.  

<AVAILABLE DOCUMENT TYPES>
{document_types}
</AVAILABLE DOCUMENT TYPES>

**Instructions:**
- Use the `document_type` field only if the query mentions one of the available document types exactly (pay attention to underscores).
- Use `document_date` for any dates or date ranges mentioned.
- Use `total_amount` for any amounts or amount ranges mentioned.
- Set `can_be_filtered` to True if at least one valid filter (document type, document_date, or total_amount) is identified; otherwise, set it to False.
- your output should be in JSON format.

**Examples of Filterable Queries:**
- "Show me all invoices from 01/01/2023 to 31/01/2023 with a total amount greater than 1000."
→ document_type: invoice  
    document_date: "01/01/2023 to 31/01/2023"  
    total_amount: "greater than 1000"  
    can_be_filtered: True

- "Retrieve purchase_order dated between 15/02/2023 and 28/02/2023 with an amount less than 500."
→ document_type: purchase_order  
    document_date: "15/02/2023 to 28/02/2023"  
    total_amount: "less than 500"  
    can_be_filtered: True

- "Find quotation received from 05/03/2023 to 10/03/2023 where the total amount is at least 2000."
→ document_type: quotation  
    document_date: "05/03/2023 to 10/03/2023"  
    total_amount: "greater than or equal to 2000"  
    can_be_filtered: True

- "Change request from James on 19/12/2018."
→ document_type: change_request  
    document_received_date: "19/12/2018"  
    can_be_filtered: True

- "Get me all resumes received from 01/01/2023 to 31/01/2023."
→ document_type: resume  
    document_received_date: "01/01/2023 to 31/01/2023"  
    can_be_filtered: True

**Examples of Non-Filterable Queries:**
- "Find documents about technology trends."
→ document_type: None  
    can_be_filtered: False

- "Retrieve the document that best explains our new policy."
→ document_type: None  
    can_be_filtered: False

- "Email from James regarding change request."
→ document_type: None  
    can_be_filtered: False
"""


class DocumentType(Enum):
    invoice = "invoice"
    purchase_order = "purchase_order"
    quotation = "quotation"
    change_request = "change_request"
    resume = "resume"
    other = "other"

class DateFilter(BaseModel):
    start_date: str = Field(description="Date format: DD/MM/YYYY")
    end_date: str = Field(description="Date format: DD/MM/YYYY")


class AmountFilter(BaseModel):
    amount: float = Field(description="Amount without currency symbol")
    greater_than: bool = Field(description="Amount > value")
    less_than: bool = Field(description="Amount < value")
    greater_than_equal: bool = Field(description="Amount >= value")
    less_than_equal: bool = Field(description="Amount <= value")

class Query(BaseModel):
    document_type: Optional[str] = Field(
        default=None,
        description="Document type to search for as provided in AVAILABLE DOCUMENT TYPES",
    )
    document_date: Optional[DateFilter] = Field(
        default=None,
        description="Date filter for document date - which is the date of the document itself",
    )
    document_received_date: Optional[DateFilter] = Field(
        default=None,
        description="Date filter for document received date - which is the date when the document was loaded into the system",
    )
    total_amount: Optional[AmountFilter] = Field(
        default=None, description="Amount filter for total amount"
    )
    can_be_filtered: bool = Field(
        description="True if query can be filtered, i.e. if it contains a document type, document date, received date or total amount"
    )
    

model = OpenAIModel(
    "qwen-plus",
    provider=OpenAIProvider(
        api_key=os.getenv("ALIBABA_API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )   
)

agent = Agent(
    model,
    # result_type=Query,
    system_prompt=SINGLE_QUERY_SYSTEM_PROMPT.format(document_types=DocumentType),
)




if __name__ == "__main__":
    query = "Show me all invoices from 01/01/2023 to 31/01/2023 with a total amount greater than 1000."
    result = asyncio.run(agent.run(query))
    print(result.data)

    # query cant be filtered
    query = "Show me all documents about technology trends."
    result = asyncio.run(agent.run(query))
    print(result.data)
    
    # query cant be filtered
    query = "get all change requests from 01/01/2023 to 31/01/2023"
    result = asyncio.run(agent.run(query))
    print(result.data)
    
    # query cant be filtered
    query = "Retrieve the document that best explains our new policy."
    result = asyncio.run(agent.run(query))
    print(result.data)
    
