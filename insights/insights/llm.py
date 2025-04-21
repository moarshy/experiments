from typing import List, Optional, Dict, Any, Union, Type, TypeVar, Generic, Protocol
from pydantic import BaseModel
import json
import time
import logging
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, 
                user_prompt: str,
                system_prompt: str = "You are a helpful assistant.",
                response_model: Optional[Type[BaseModel]] = None,
                temperature: float = 0.0,
                max_tokens: Optional[int] = None,
                **kwargs) -> Union[str, BaseModel]:
        """Generate a response from the LLM, optionally parsing into a Pydantic model."""
        pass


class OpenAIClient(BaseLLMClient):
    """Unified client for interacting with OpenAI API with support for both structured and unstructured output."""
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the OpenAI client with default model.
        
        Args:
            model: The OpenAI model to use
        """
        try:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = model
        except ImportError:
            raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'")
    
    def generate(self, 
                user_prompt: str,
                system_prompt: str = "You are a helpful assistant.",
                response_model: Optional[Type[BaseModel]] = None,
                temperature: float = 0.0,
                max_tokens: Optional[int] = None,
                max_retries: int = 3,
                **kwargs) -> Union[str, BaseModel]:
        """
        Generate a response from the OpenAI API, optionally parsing into a Pydantic model.
        
        Args:
            user_prompt: The user prompt to send to the API
            system_prompt: The system prompt to set context
            response_model: Optional Pydantic model class to parse the response into
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            max_retries: Number of retry attempts on failure
            
        Returns:
            Either a string (unstructured) or an instance of the specified Pydantic model (structured)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for i in range(max_retries):
            try:
                # Handle structured output with Pydantic model
                if response_model is not None:
                    schema = response_model.model_json_schema()
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=[{"type": "function", "function": {"name": "generate_structured_output", "parameters": schema}}],
                        tool_choice={"type": "function", "function": {"name": "generate_structured_output"}}
                    )
                    function_call = response.choices[0].message.tool_calls[0].function
                    return response_model.model_validate(json.loads(function_call.arguments))
                
                # Handle normal text output
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                if i < max_retries - 1:
                    logger.info(f"Retrying... ({i + 1}/{max_retries})")
                    time.sleep(2)
                    messages.append({"role": "user", "content": f"Please try again. Here is the error: {str(e)}"})
                    continue
                else:
                    logger.error(f"Failed to call OpenAI after {max_retries} retries: {str(e)}")
                    raise e
                    
        raise ValueError("Failed to generate output")


class GeminiClient(BaseLLMClient):
    """Unified client for interacting with Gemini API with support for both structured and unstructured output."""
    
    def __init__(self, model: str = "models/gemini-2.5-flash-preview-04-17"):
        """
        Initialize the Gemini client with default model.
        
        Args:
            model: The Gemini model to use
        """
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client()
            self.model = model
            self.types = types
        except ImportError:
            raise ImportError("Google Generative AI package is not installed. Install it with 'pip install google-generativeai'")
    
    def generate(self, 
                user_prompt: str,
                system_prompt: str = "You are a helpful assistant.",
                response_model: Optional[Type[BaseModel]] = None,
                temperature: float = 0.0,
                max_tokens: Optional[int] = None,
                thinking_budget: Optional[int] = None,
                max_retries: int = 3,
                **kwargs) -> Union[str, BaseModel]:
        """
        Generate a response from the Gemini API, optionally parsing into a Pydantic model.
        
        Args:
            user_prompt: The user prompt to send to the API
            system_prompt: The system prompt to set context
            response_model: Optional Pydantic model class to parse the response into
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            thinking_budget: Maximum number of tokens for thinking
            max_retries: Number of retry attempts on failure
            
        Returns:
            Either a string (unstructured) or an instance of the specified Pydantic model (structured)
        """
        for i in range(max_retries):
            try:
                # Handle structured output with Pydantic model
                if response_model is not None:
                    config = self.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        system_instruction=system_prompt,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget else None,
                        response_mime_type="application/json",
                        response_schema=response_model,
                    )
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[user_prompt],
                        config=config
                    )
                    return response_model.model_validate(json.loads(response.candidates[0].content.parts[0].text))

                # Handle normal text output
                else:
                    config = self.types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        system_instruction=system_prompt,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget else None,
                    )

                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[user_prompt],
                        config=config
                    )

                    return response.candidates[0].content.parts[0].text
                    
            except Exception as e:
                if i < max_retries - 1:
                    logger.info(f"Retrying... ({i + 1}/{max_retries})")
                    user_prompt += f"Please try again. Here is the error: {str(e)}"
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Failed to call Gemini after {max_retries} retries: {str(e)}")
                    raise e
                    
        raise ValueError("Failed to generate output")


class LLM:
    """Factory for creating LLM clients with a unified interface."""
    
    @staticmethod
    def create_client(provider: str = "openai", **kwargs) -> BaseLLMClient:
        """
        Create a client for the specified provider.
        
        Args:
            provider: The LLM provider to use ('openai' or 'gemini')
            **kwargs: Additional arguments to pass to the client constructor
            
        Returns:
            An instance of BaseLLMClient for the specified provider
        """
        if provider.lower() == "openai":
            return OpenAIClient(**kwargs)
        elif provider.lower() in ["gemini", "google"]:
            return GeminiClient(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'openai', 'gemini'")


# Example usage
if __name__ == "__main__":
    from pydantic import Field
    from dotenv import load_dotenv
    load_dotenv()
    
    # Define a Pydantic model for structured output
    class MovieReview(BaseModel):
        title: str = Field(description="The title of the movie")
        year: int = Field(description="The year the movie was released")
        rating: float = Field(description="Rating from 0.0 to 10.0")
        summary: str = Field(description="A brief summary of the review")
    
    # Create an OpenAI client
    openai_client = LLM.create_client("openai", model="gpt-4.1")
    
    # Create a Gemini client
    gemini_client = LLM.create_client("gemini", model="models/gemini-2.5-flash-preview-04-17")
    
    # Example with OpenAI
    review = openai_client.generate(
        user_prompt="Write a review of the movie 'Inception'",
        response_model=MovieReview
    )
    print(review)

    # Example with OpenAI simple text output
    text = openai_client.generate(
        user_prompt="Write a short poem about AI",
        system_prompt="You are a creative poet."
    )
    print(text)

    # Example with Gemini structured output 
    review = gemini_client.generate(
        user_prompt="Write a review of the movie 'Inception'",
        response_model=MovieReview
    )
    print(review)
    
    # Example with Gemini simple text output
    text = gemini_client.generate(
        user_prompt="Write a short poem about AI",
        system_prompt="You are a creative poet."
    )
    print(text)