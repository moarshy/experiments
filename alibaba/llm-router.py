import asyncio
import json
import re
from enum import Enum
import os
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ALIBABA_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


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
- For dates, extract as DateFilter objects with start_date and end_date fields (both in DD/MM/YYYY format).
- For amounts, extract as AmountFilter objects with amount (numeric value) and appropriate boolean flags.
- Set `can_be_filtered` to True if at least one valid filter (document type, document_date, or total_amount) is identified; otherwise, set it to False.
- Your output should be in JSON format.

**Examples of Filterable Queries:**
- "Show me all invoices from 01/01/2023 to 31/01/2023 with a total amount greater than 1000."
```json
{{
  "document_type": "invoice",
  "document_date": {{
    "start_date": "01/01/2023",
    "end_date": "31/01/2023"
  }},
  "total_amount": {{
    "amount": 1000,
    "greater_than": true,
    "less_than": false,
    "greater_than_equal": false,
    "less_than_equal": false
  }},
  "can_be_filtered": true
}}
```

- "Retrieve purchase_order dated between 15/02/2023 and 28/02/2023 with an amount less than 500."
```json
{{
  "document_type": "purchase_order",
  "document_date": {{
    "start_date": "15/02/2023",
    "end_date": "28/02/2023"
  }},
  "total_amount": {{
    "amount": 500,
    "greater_than": false,
    "less_than": true,
    "greater_than_equal": false,
    "less_than_equal": false
  }},
  "can_be_filtered": true
}}
```

- "Find quotation received from 05/03/2023 to 10/03/2023 where the total amount is at least 2000."
```json
{{
  "document_type": "quotation",
  "document_received_date": {{
    "start_date": "05/03/2023",
    "end_date": "10/03/2023"
  }},
  "total_amount": {{
    "amount": 2000,
    "greater_than": false,
    "less_than": false,
    "greater_than_equal": true,
    "less_than_equal": false
  }},
  "can_be_filtered": true
}}
```

- "Change request from James on 19/12/2018."
```json
{{
  "document_type": "change_request",
  "document_received_date": {{
    "start_date": "19/12/2018",
    "end_date": "19/12/2018"
  }},
  "can_be_filtered": true
}}
```

- "Get me all resumes received from 01/01/2023 to 31/01/2023."
```json
{{
  "document_type": "resume",
  "document_received_date": {{
    "start_date": "01/01/2023",
    "end_date": "31/01/2023"
  }},
  "can_be_filtered": true
}}
```

**Examples of Non-Filterable Queries:**
- "Find documents about technology trends."
```json
{{
  "document_type": null,
  "document_date": null,
  "document_received_date": null,
  "total_amount": null,
  "can_be_filtered": false
}}
```

- "Retrieve the document that best explains our new policy."
```json
{{
  "document_type": null,
  "document_date": null,
  "document_received_date": null,
  "total_amount": null,
  "can_be_filtered": false
}}
```

- "Email from James regarding change request."
```json
{{
  "document_type": null,
  "document_date": null,
  "document_received_date": null,
  "total_amount": null,
  "can_be_filtered": false
}}
```
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

def extract_json_from_response(response_text: str) -> str:
    """Extract JSON content from the response text."""
    # Try to find JSON within code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
    if json_match:
        return json_match.group(1).strip()
    
    # If no code blocks, try to find JSON-like content
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if json_match:
        return json_match.group(0).strip()
    
    return response_text

def parse_date_range(date_str: str) -> DateFilter:
    """Parse a date range string into a DateFilter object."""
    if ' to ' in date_str:
        start_date, end_date = date_str.split(' to ')
        return DateFilter(start_date=start_date.strip(), end_date=end_date.strip())
    else:
        # Assume it's a single date
        return DateFilter(start_date=date_str.strip(), end_date=date_str.strip())

def parse_amount_filter(amount_str: str) -> AmountFilter:
    """Parse an amount filter string into an AmountFilter object."""
    amount_value = 0.0
    greater_than = False
    less_than = False
    greater_than_equal = False
    less_than_equal = False
    
    # Extract the numeric value
    numeric_match = re.search(r'(\d+(?:\.\d+)?)', amount_str)
    if numeric_match:
        amount_value = float(numeric_match.group(1))
    
    # Determine comparison type
    if "greater than or equal" in amount_str or "at least" in amount_str:
        greater_than_equal = True
    elif "less than or equal" in amount_str or "at most" in amount_str:
        less_than_equal = True
    elif "greater than" in amount_str:
        greater_than = True
    elif "less than" in amount_str:
        less_than = True
    
    return AmountFilter(
        amount=amount_value,
        greater_than=greater_than,
        less_than=less_than,
        greater_than_equal=greater_than_equal,
        less_than_equal=less_than_equal
    )

def process_user_query(query_text: str, max_retries: int = 3) -> Query:
    """
    Process a user query and return a structured Query object.
    Uses a chat-based approach to correct JSON errors.
    """
    # Initial system prompt with document types
    system_prompt = SINGLE_QUERY_SYSTEM_PROMPT.format(
        document_types="\n".join([dt.value for dt in DocumentType])
    )
    
    # Initialize conversation history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_text}
    ]
    
    # First attempt
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )
    
    response_content = response.choices[0].message.content
    print(f"Initial response: {response_content}")
    
    # Extract JSON from the response
    json_content = extract_json_from_response(response_content)
    
    # Try to parse the JSON
    for attempt in range(max_retries):
        try:
            json_data = json.loads(json_content)
            
            # Convert string representations to proper objects if needed
            if isinstance(json_data.get("document_date"), str) and json_data["document_date"]:
                json_data["document_date"] = parse_date_range(json_data["document_date"])
            
            if isinstance(json_data.get("document_received_date"), str) and json_data["document_received_date"]:
                json_data["document_received_date"] = parse_date_range(json_data["document_received_date"])
                
            if isinstance(json_data.get("total_amount"), str) and json_data["total_amount"]:
                json_data["total_amount"] = parse_amount_filter(json_data["total_amount"])
            
            return Query(**json_data)
        
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")
            
            # Add the response to conversation history
            messages.append({"role": "assistant", "content": response_content})
            
            # Add error correction request to conversation history
            error_message = (
                f"There was an error parsing your JSON response: {str(e)}. "
                f"Please provide a corrected JSON that follows this structure:\n"
                "{\n"
                '  "document_type": string or null,\n'
                '  "document_date": {\n'
                '    "start_date": "DD/MM/YYYY",\n'
                '    "end_date": "DD/MM/YYYY"\n'
                '  } or null,\n'
                '  "document_received_date": {\n'
                '    "start_date": "DD/MM/YYYY",\n'
                '    "end_date": "DD/MM/YYYY"\n'
                '  } or null,\n'
                '  "total_amount": {\n'
                '    "amount": float,\n'
                '    "greater_than": boolean,\n'
                '    "less_than": boolean,\n'
                '    "greater_than_equal": boolean,\n'
                '    "less_than_equal": boolean\n'
                '  } or null,\n'
                '  "can_be_filtered": boolean\n'
                "}\n"
                "Return only the corrected JSON with no explanation."
            )
            messages.append({"role": "user", "content": error_message})
            
            # Get corrected response
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=messages
            )
            
            response_content = response.choices[0].message.content
            print(f"Correction attempt {attempt+1}: {response_content}")
            
            # Extract JSON from the response
            json_content = extract_json_from_response(response_content)
    
    # If we've reached here, all attempts failed
    print(f"Failed to parse after {max_retries} attempts.")
    return Query(can_be_filtered=False)

# Test queries - expanded to include more edge cases
test_queries = [
    # Basic document type queries
    "Show me all invoices from 01/01/2023 to 31/01/2023 with a total amount greater than 1000.",
    "Find all purchase_order documents.",
    "Get me quotation documents.",
    "List all change_request documents.",
    "Search for resume files.",
    
    # Date formats and variations
    "Invoices dated 15/06/2022.",
    "Quotations from January 2023.",
    "Purchase orders between 01/03/2023 and 30/04/2023.",
    "Resumes received after 15/07/2022.",
    "Change requests before 31/12/2022.",
    
    # Amount variations
    "Invoices with amount greater than 5000.",
    "Purchase orders with total less than 250.",
    "Quotations with price at least 1500.",
    "Invoices with amount at most 3000.",
    "Purchase orders with total between 1000 and 2000.",
    
    # Combinations
    "Invoices from March 2023 with amount over 750.",
    "Purchase orders received between 01/05/2023 and 31/05/2023 with total under 1500.",
    "Quotations dated 10/02/2023 with amount exactly 2250.",
    "Resumes received in 2022 Q4.",
    "Change requests from James with amount greater than or equal to 500.",
    
    # Non-filterable queries
    "Find documents about technology trends.",
    "Retrieve the document that best explains our new policy.",
    "Email from James regarding project updates.",
    "The latest documentation for our product.",
    "Give me everything about marketing campaigns.",
    
    # Edge cases with partial or ambiguous information
    "I need something about invoices.",
    "Documents with date 01/01/2023.",
    "Files with amount 1000.",
    "Change request files but I don't know the date.",
    "Something from James from yesterday."
]

if __name__ == "__main__":
    # Process a subset or all test queries
    for i, query in enumerate(test_queries):
        print(f"\nProcessing query {i+1}/{len(test_queries)}: '{query}'")
        result = process_user_query(query)
        print(f"Structured result: {result.model_dump()}")