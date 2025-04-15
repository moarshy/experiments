import logging
import json
import time
from openai import OpenAI, RateLimitError, APIError
from insights.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from insights.utils import setup_logging
# >>> Step 1: Add this import <<<
from pydantic import BaseModel, ValidationError, TypeAdapter
from typing import Optional, TypeVar, Type, Union, cast, List, Any

logger = setup_logging()

# Ensure client initialization happens correctly
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.exception(f"Failed to initialize OpenAI client: {e}")
    raise

T = TypeVar('T') # Generic type for return value

def call_openai_api(
    system_prompt: str,
    user_prompt: str,
    response_model: Optional[Type[Any]] = None, # Accept relevant type hints
    model: str = OPENAI_MODEL_NAME,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    max_retries: int = 3,
) -> Union[str, T, None]:
    """
    Calls the OpenAI API, parsing the response into the specified Pydantic model,
    list of models, or returning raw text. Simplest approach using TypeAdapter.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format_config = {"type": "json_object"} if response_model else None
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Calling OpenAI API (Attempt {attempt + 1}/{max_retries})...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format_config
            )
            content = response.choices[0].message.content
            if not content:
                logger.warning(f"OpenAI returned empty content (Attempt {attempt+1})")
                last_exception = ValueError("OpenAI returned empty content")
                if attempt < max_retries - 1: time.sleep(1); continue
                else: return None

            logger.info(f"OpenAI raw response (Attempt {attempt+1}): {content}")

            if not response_model:
                return content.strip() # Return raw text

            try:
                # >>> Step 2: Use TypeAdapter for validation <<<
                adapter = TypeAdapter(response_model)
                parsed_response = adapter.validate_json(content) # Handles single/list/etc.
                logger.info("Successfully parsed and validated response.")
                return cast(T, parsed_response) # Return the validated data

            except (json.JSONDecodeError, ValidationError) as validation_error:
                last_exception = validation_error
                error_detail = f"{type(validation_error).__name__}: {validation_error}"
                # Try to get schema for error message (TypeAdapter helps here)
                try:
                    model_schema_json = json.dumps(adapter.json_schema(), indent=2)
                except Exception: model_schema_json = f"Expected format based on type hint: {response_model}"

                if attempt < max_retries - 1:
                    logger.warning(f"Response validation failed (Attempt {attempt+1}): {error_detail}")
                    error_message = (
                        f"The previous response could not be parsed/validated.\n"
                        f"--- Error ---\n{error_detail}\n"
                        f"--- Previous Response ---\n{content}\n--- End Previous Response ---\n"
                        f"Please ensure your *entire* output is a single JSON structure matching this schema:\n"
                        f"{model_schema_json}"
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": error_message})
                    response_format_config = {"type": "json_object"} # Re-ensure JSON mode
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Failed to validate response after {max_retries} attempts: {error_detail}")
                    logger.error(f"Final failing content: {content}")
                    return None # Failed after all retries

        # (Keep existing RateLimitError, APIError, Exception handling as before)
        except RateLimitError as e: last_exception = e; logger.warning(...); # Handle retry/backoff
        except APIError as e: last_exception = e; logger.error(...); # Handle retry/fail
        except Exception as e: last_exception = e; logger.exception(...); return None # Fail on unexpected errors

    logger.error(f"Exited retry loop unexpectedly. Last known error: {last_exception}")
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
