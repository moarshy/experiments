"""
OpenAI client with combined structure and summary generation for trading applications
"""
import json
import logging
from typing import Dict, Any, List, Optional

import httpx

from research_trader.config import settings
from research_trader.utils.errors import ServiceError

# Configure logging
logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for interacting with the OpenAI API"""
    
    def __init__(self, api_key: str = settings.OPENAI_API_KEY, model: str = settings.OPENAI_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
    
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to OpenAI API
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            API response as dictionary
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=settings.OPENAI_TIMEOUT) as client:
                response = await client.post(
                    f"{self.base_url}/{endpoint}",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                logger.info(f"Response: {response.json()}")
                return response.json()
        except httpx.HTTPStatusError as e:
            error_message = f"OpenAI API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                if "error" in error_data and "message" in error_data["error"]:
                    error_message = f"OpenAI API error: {error_data['error']['message']}"
            except ValueError:
                pass
                
            raise ServiceError(
                detail=error_message,
                service="openai",
                status_code=503
            )
        except httpx.RequestError as e:
            raise ServiceError(
                detail=f"Request error: {str(e)}",
                service="openai",
                status_code=503
            )
    
    async def generate_paper_summary_and_structure(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combined method to extract structure and generate summary in one call
        
        Args:
            paper_data: Dictionary with paper information
                - id: Paper ID
                - title: Paper title
                - authors: List of authors
                - text: Paper text (abstract or full text)
                - is_full_text: Whether text is the full paper or just abstract
            
        Returns:
            Dictionary with structure and summary information
        """
        # Format author list
        authors_str = ", ".join(paper_data["authors"]) if "authors" in paper_data else "Unknown"
        
        # Different prompt based on whether we have full text or just abstract
        if paper_data.get("is_full_text", False):
            system_prompt = """You are a specialized AI for analyzing quantitative finance and trading research papers.
            You are provided with the FULL TEXT of a paper. Extract key information and create a comprehensive analysis for quantitative traders.
            
            Provide the following:
            
            1. objective: Concise statement of the paper's main goal (single string, not a list)
            2. methods: List of specific technical approaches, algorithms, or models used (3-5 specific items)
            3. results: List of key quantitative and qualitative findings, especially performance metrics (3-5 specific items)
            4. conclusions: List of main takeaways and implications for trading (3-5 specific items)
            5. trading_applications: List of concrete ways this research could be applied to actual trading systems (3-5 specific items)
            6. summary: A concise overview of the paper (150-200 words)
            7. keywords: List of 5-8 specific, technical keywords relevant to trading
            8. implementation_complexity: Assessment of how difficult it would be to implement (Low/Medium/High) with brief justification
            9. data_requirements: List of specific data types and sources needed to implement the approach (3-5 items)
            10. potential_performance: Assessment of expected trading performance based on the paper's results
            
            Format your response as a JSON object with these keys.
            Focus on extracting actionable insights for quant traders and practical implementation details."""
        else:
            system_prompt = """You are a specialized AI for analyzing quantitative finance and trading research papers.
            You are provided with the ABSTRACT of a paper. Extract key information and create a comprehensive analysis for quantitative traders.
            
            Provide the following (infer as much as possible from the limited information):
            
            1. objective: Concise statement of the paper's main goal (single string, not a list)
            2. methods: List of specific technical approaches, algorithms, or models used (2-4 specific items)
            3. results: List of key quantitative and qualitative findings, especially performance metrics (2-4 specific items)
            4. conclusions: List of main takeaways and implications for trading (2-4 specific items)
            5. trading_applications: List of potential ways this research could be applied to trading systems (2-4 specific items)
            6. summary: A concise overview of the paper (100-150 words)
            7. keywords: List of 4-6 specific, technical keywords likely relevant to the paper
            8. implementation_complexity: Assessment of how difficult it would be to implement (Low/Medium/High) with brief justification
            9. data_requirements: List of likely data types and sources needed to implement the approach (2-4 items)
            10. potential_performance: Assessment of potential trading performance based on the limited information
            
            Format your response as a JSON object with these keys.
            Be clear when you're making educated guesses due to limited information from just the abstract."""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Paper to analyze:
                
                Title: {paper_data['title']}
                Authors: {authors_str}
                
                {'Full Text' if paper_data.get('is_full_text', False) else 'Abstract'}:
                {paper_data['text']}
                """
            }
        ]
        
        response = await self._make_request(
            "chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            result_data = json.loads(content)
            logger.info(f"Structure and summary: {result_data}")
            
            # Normalize data and ensure required fields
            normalized_data = {
                "objective": "No clear objective found in the paper.",
                "methods": [],
                "results": [],
                "conclusions": [],
                "trading_applications": [],
                "summary": "Summary not available.",
                "keywords": [],
                "implementation_complexity": "Medium",
                "data_requirements": [],
                "potential_performance": "Not specified"
            }
            
            # Process each field, handling both string and list formats
            for key in normalized_data.keys():
                # Check both lowercase and capitalized versions
                for check_key in [key, key.capitalize()]:
                    if check_key in result_data:
                        # String fields
                        if key in ["objective", "summary", "implementation_complexity", "potential_performance"]:
                            if isinstance(result_data[check_key], str):
                                normalized_data[key] = result_data[check_key]
                                break
                        
                        # List fields
                        elif key in ["methods", "results", "conclusions", "trading_applications", "keywords", "data_requirements"]:
                            if isinstance(result_data[check_key], list):
                                normalized_data[key] = result_data[check_key]
                                break
                            elif isinstance(result_data[check_key], str):
                                # Try to parse string as a list
                                text = result_data[check_key]
                                
                                # Split on common list patterns
                                if "\n-" in text or "\n*" in text or "\n•" in text:
                                    items = []
                                    for line in text.split("\n"):
                                        line = line.strip()
                                        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
                                            items.append(line[2:])
                                    if items:
                                        normalized_data[key] = items
                                        break
                                
                                # Try comma-separated items
                                items = [item.strip() for item in text.split(",") if item.strip()]
                                if items:
                                    normalized_data[key] = items
                                    break
            
            return normalized_data
                    
        except (KeyError, json.JSONDecodeError) as e:
            raise ServiceError(
                detail=f"Failed to process paper: {str(e)}",
                service="openai",
                status_code=500
            )
    
    async def answer_question(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Answer a question based on the provided context
        
        Args:
            question: User's question
            context: List of paper summaries and structured information
            
        Returns:
            Dictionary with the answer and sources
        """
        # Prepare the context
        formatted_context = []
        for idx, paper in enumerate(context):
            # Format methods, results, etc. as bullet points
            methods_str = "\n- " + "\n- ".join(paper.get("methods", [])) if paper.get("methods") else "Not specified"
            results_str = "\n- " + "\n- ".join(paper.get("results", [])) if paper.get("results") else "Not specified"
            conclusions_str = "\n- " + "\n- ".join(paper.get("conclusions", [])) if paper.get("conclusions") else "Not specified"
            trading_applications_str = "\n- " + "\n- ".join(paper.get("trading_applications", [])) if paper.get("trading_applications") else "Not specified"
            
            formatted_context.append(
                f"Paper {idx+1}: {paper.get('title', 'Untitled')}\n"
                f"Authors: {', '.join(paper.get('authors', ['Unknown'])) if 'authors' in paper else 'Unknown'}\n"
                f"Summary: {paper.get('summary', 'No summary available')}\n"
                f"Objective: {paper.get('objective', 'Not specified')}\n"
                f"Methods: {methods_str}\n"
                f"Results: {results_str}\n"
                f"Conclusions: {conclusions_str}\n"
                f"Trading Applications: {trading_applications_str}\n"
            )
        
        context_text = "\n\n".join(formatted_context)
        
        messages = [
            {
                "role": "system",
                "content": """You are a research assistant specialized in quantitative finance and trading strategies.
                Answer questions based ONLY on the provided research paper information.
                If the answer cannot be found in the provided context, say so clearly.
                Cite your sources by referring to the Paper number (e.g., "According to Paper 2...").
                
                Focus on practical, actionable insights for traders when appropriate.
                Suggest follow-up questions or areas to explore after your answer.
                
                Format your response as a JSON object with these keys:
                - "answer": your detailed answer to the question
                - "sources": list of paper numbers used as sources (e.g., [1, 3])
                - "confidence": a score between 0 and 1 indicating your confidence in the answer
                - "suggestions": list of 2-3 follow-up questions or areas to explore"""
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}"
            }
        ]
        
        response = await self._make_request(
            "chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
        )
        
        try:
            content = response["choices"][0]["message"]["content"]
            answer_data = json.loads(content)
            
            # Ensure proper format
            result = {
                "question": question,
                "answer": answer_data.get("answer", "Unable to answer the question based on the provided context."),
                "sources": answer_data.get("sources", []),
                "confidence": answer_data.get("confidence", 0.0),
                "suggestions": answer_data.get("suggestions", [])
            }
                    
            return result
        except (KeyError, json.JSONDecodeError) as e:
            raise ServiceError(
                detail=f"Failed to generate answer: {str(e)}",
                service="openai",
                status_code=500
            )