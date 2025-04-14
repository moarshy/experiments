import logging
import json
from openai import OpenAI, RateLimitError, APIError
from insights.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from pydantic import BaseModel, ValidationError
from typing import Optional, TypeVar, Generic, Type, Union, cast

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

T = TypeVar('T', bound=BaseModel)

def call_openai_api(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[T]] = None,
    model: str = OPENAI_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    max_retries: int = 3,
) -> Union[str, T, None]:
    """
    Calls the OpenAI API with the specified prompt and parameters.

    Args:
        system_prompt: The system prompt to send to the LLM.
        user_prompt: The user prompt to send to the LLM.
        response_model: Optional Pydantic model to validate and parse the response.
        model: The model name to use.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        max_retries: Maximum number of retries for response model validation.

    Returns:
        If response_model is provided, returns the parsed model instance.
        Otherwise, returns the raw text response from the LLM.
        Returns None if an error occurs and cannot be recovered.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if response_model else None
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning("OpenAI returned empty content")
                return None
                
            logger.debug(f"OpenAI raw response (attempt {attempt+1}): {content}")
            
            # If no response model is expected, return the raw content
            if not response_model:
                return content.strip()
            
            # Try to parse and validate against the response model
            try:
                # First ensure we have valid JSON
                json_data = json.loads(content)
                # Then validate with Pydantic
                parsed_response = response_model.model_validate(json_data)
                return parsed_response
                
            except (json.JSONDecodeError, ValidationError) as validation_error:
                if attempt < max_retries - 1:
                    logger.warning(f"Response validation failed (attempt {attempt+1}): {validation_error}")
                    
                    # Add correction message for the next attempt
                    error_message = (
                        f"The previous response could not be parsed into the required format. "
                        f"Error: {str(validation_error)}. "
                        f"Please correct your response to match this exact schema: {response_model.schema_json()}"
                    )
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": error_message})
                    continue
                else:
                    logger.error(f"Failed to validate response after {max_retries} attempts: {validation_error}")
                    return None
                    
        except RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error: {e}")
            # Implement backoff strategy if needed
            return None
        except APIError as e:
            logger.error(f"OpenAI API Error: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred calling OpenAI: {e}")
            return None
    
    return None
    
if __name__ == "__main__":
    from pydantic import BaseModel, Field
    
    class PersonInfo(BaseModel):
        name: str
        age: int
        gender: str
    
    system_prompt = "You are a helpful assistant designed to output JSON. Extract name, age, and gender from the following text: "
    user_prompt = "John Doe is 30 years old and is a male."
    
    # Test with response model
    result = call_openai_api(system_prompt, user_prompt, response_model=PersonInfo)
    print(f"Parsed result: {result}")
    
    # Test without response model
    raw_result = call_openai_api(system_prompt, user_prompt)
    print(f"Raw result: {raw_result}")
