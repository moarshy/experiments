import time
import json
import re
from typing import List, Optional
from openai import OpenAI
from google import genai
from google.genai import types

from insights.utils import setup_logging

logger = setup_logging()

import re

def parse_json(text):
    """
    Robustly extracts JSON from text that might contain non-JSON content.
    
    Args:
        text (str): Text that might contain JSON data
        
    Returns:
        dict or list: Parsed JSON data
        None: If no valid JSON found
    """
    # First, try to extract JSON-like patterns from text
    # Look for content between curly braces (for objects) or square brackets (for arrays)
    json_pattern = re.compile(r'({[\s\S]*?}|\[[\s\S]*?\])')
    
    # Try various parsing approaches
    
    # Approach 1: Try parsing the entire text directly
    try:
        return json.loads(text)
    except:
        pass
    
    # Approach 2: Look for markdown code blocks with json
    code_block_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
    code_matches = code_block_pattern.findall(text)
    
    for code_match in code_matches:
        try:
            return json.loads(code_match)
        except:
            pass
    
    # Approach 3: Extract JSON-like patterns and try to parse them
    json_matches = json_pattern.findall(text)
    
    for json_match in json_matches:
        try:
            return json.loads(json_match)
        except:
            pass
    
    # Approach 4: Clean up the text and try again
    # Remove common formatting issues
    for pattern in [r'`', r'"', r'"', r''', r''']: 
        text = text.replace(pattern, '"')
    
    # Replace single quotes with double quotes for JSON compatibility
    # This is risky but works in many cases
    text_cleaned = text.replace("'", '"')
    
    try:
        return json.loads(text_cleaned)
    except:
        pass
        
    # Approach 5: Try to find a substring that looks like JSON
    for i in range(len(text)):
        if text[i] in ['{', '[']:
            # Try to parse everything from this point
            try:
                return json.loads(text[i:])
            except:
                pass
    
    # If all attempts fail
    return None

def openai_call(
    user_prompt: str,
    system_prompt: str,
    model: str = 'gpt-4.1',
    temperature: float = 0.4,
    max_tokens: int = 10000,
    max_retries: int = 3,
    expected_keys: Optional[List[str]] = None,
    ):

    client = OpenAI()

    for i in range(max_retries):
        try:
            if expected_keys:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                response_json = json.loads(response.choices[0].message.content)
                if all(key in response_json for key in expected_keys):
                    return response_json
                else:
                    raise ValueError(f"Expected keys {expected_keys} not found in response")
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            if i < max_retries - 1:
                logger.info(f"Retrying... ({i + 1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                logger.error(f"Failed to call OpenAI after {max_retries} retries")
                raise e

def gemini_call(
    user_prompt: str,
    system_prompt: str,
    model: str = 'models/gemini-2.5-flash-preview-04-17',
    temperature: float = 0.2,
    max_tokens: int = 10000,
    thinking_budget: int = 10000,
    max_retries: int = 3,
    expected_keys: Optional[List[str]] = None,
    ):

    client = genai.Client()

    for i in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            )
            response = client.models.generate_content(
                model=model,
                contents=[user_prompt],
                config=config
            )
            response_text = response.candidates[0].content.parts[0].text
            try:
                response_json = parse_json(response_text)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON. {e}")

            if expected_keys:
                if all(key in response_json for key in expected_keys):
                    return response_json
                else:
                    raise ValueError(f"Expected keys {expected_keys} not found in response")
            return response_json
            
        except Exception as e:
            if i < max_retries - 1:
                logger.info(f"Retrying... ({i + 1}/{max_retries})")
                time.sleep(2)
                continue
            else:
                logger.error(f"Failed to call OpenAI after {max_retries} retries")
                raise e
