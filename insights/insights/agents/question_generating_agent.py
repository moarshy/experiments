# insights/question_generating_agent.py

import json
import re
import uuid
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field, field_validator

from insights.llm import openai_call, gemini_call
from insights.utils import setup_logging 

logger = setup_logging()

# --- Pydantic Models ---
class LLMResponse(BaseModel):
    """Represents the response from an LLM."""
    questions: List[str]

class AnalysisQuestion(BaseModel):
    """Represents a single generated analytical question."""
    question_id: str = Field(default_factory=lambda: f"Q-{uuid.uuid4()}")
    question_text: str
    source_llm: str
    iteration_level: int

    @field_validator('question_text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Question text must not be empty')
        return v.strip()


class GeneratedQuestions(BaseModel):
    """Represents the final output of the generator."""
    questions: List[AnalysisQuestion] = Field(default_factory=list)
    models_used: List[str] = Field(default_factory=list)
    total_iterations_run_per_model: Dict[str, int] = Field(default_factory=dict) # Track per model
    final_question_count: int = 0


class GeneratorConfig(BaseModel):
    """Configuration for the IterativeBusinessQuestionGenerator."""
    llm_list: List[str] = ["gpt-4.1", "gemini-2.5-flash-preview-04-17"]
    max_iterations_per_model: int = 4 # Default number of loops per model
    num_questions_per_iteration_per_model: int = 3 # Target questions per call
    llm_temperature: float = 0.4 # Default LLM temperature
    llm_max_output_tokens: int = 10000 # Limit output size for question lists


# --- Core Agent Class ---

class IterativeBusinessQuestionGenerator:
    """
    Generates analytical questions iteratively using multiple LLMs.
    """
    def __init__(self, config: GeneratorConfig):
        """
        Initializes the generator with the given configuration.

        Args:
            config: A GeneratorConfig object.
        """
        self.config = config
        self.all_questions: List[AnalysisQuestion] = [] # Stores all generated questions
        self._validate_config()
        logger.info(f"Initialized IterativeBusinessQuestionGenerator with config: {config.model_dump_json()}")

    def _validate_config(self):
        """Basic validation for the configuration."""
        if not self.config.llm_list:
            raise ValueError("llm_list cannot be empty.")
        if self.config.max_iterations_per_model <= 0:
            raise ValueError("max_iterations_per_model must be positive.")
        if self.config.num_questions_per_iteration_per_model <= 0:
            raise ValueError("num_questions_per_iteration_per_model must be positive.")
        
    def _response_parser(self, response: List[str], source_llm: str, iteration_level: int) -> List[AnalysisQuestion]:
        """Parse the response from the LLM into a list of AnalysisQuestion objects."""
        questions = []
        for question in response:
            question = re.sub(r'^\d+\.\s*', '', question)
            questions.append(AnalysisQuestion(question_text=question, source_llm=source_llm, iteration_level=iteration_level))
        return questions

    def _build_prompt(
        self,
        db_summary: Union[str, Dict],
        current_questions: List[AnalysisQuestion],
        target_model: str,
        iteration_num: int,
        num_questions_to_generate: int, # Use config value passed down
        user_query: Optional[str] = None
    ) -> tuple[Optional[str], str]:
        """
        Constructs the system and user prompts for the LLM.
        Instructs LLM to provide a numbered list.

        Args:
            db_summary: The database summary (string or dict).
            current_questions: List of questions generated so far.
            target_model: The LLM being prompted.
            iteration_num: The current iteration number (0-based).
            num_questions_to_generate: How many new questions to ask for.
            user_query: Optional initial query from the user.

        Returns:
            A tuple containing (system_prompt, user_prompt).
        """
        system_prompt = """
You are an expert Data Analyst and Business Strategist. Your goal is to generate insightful, complex, and actionable analytical questions about a dataset based on its summary.
Focus on questions that uncover business insights, trends, opportunities, risks, or performance drivers.
Generate questions that require non-trivial analysis (e.g., correlations, multi-step calculations, comparisons across segments, trend analysis over time) rather than simple lookups.
Ensure the questions are directly relevant to the provided database summary and the optional user query.

**Output Format Constraint:**
Structure your response strictly as a numbered list of questions with the following keys only: questions
Generate exactly """ + str(num_questions_to_generate) + """ questions.
Do not include any preamble, commentary, or concluding remarks. Just the numbered questions.

Example of expected output format:
```json
{
    "questions": [
        "1. What is the trend in X over the last Y months?",
        "2. How does metric A compare between segment B and segment C?",
        "3. Is there a correlation between feature D and outcome E?"
    ]
}
```
"""
        # Format database summary
        summary_str = ""
        if isinstance(db_summary, dict):
            summary_str_content = json.dumps(db_summary, indent=2)
            summary_str = f"Database Technical Summary:\n```json\n{summary_str_content}\n```"
        else:
            summary_str_content = str(db_summary)
            summary_str = f"Database Summary:\n{summary_str_content}"

        # Format previous questions
        previous_questions_str = "Context: Previously Generated Questions (Consider these to generate NEW diverse questions):\n"
        if current_questions:
            for i, q in enumerate(current_questions):
                previous_questions_str += f"- {q.question_text}\n" # Simple bullet points for context
        else:
            previous_questions_str += "None yet.\n"

        # Build user prompt
        user_prompt_parts = [summary_str]
        if user_query:
            user_prompt_parts.append(f"Initial User Query/Topic:\n\"{user_query}\"")

        user_prompt_parts.append(previous_questions_str)
        user_prompt_parts.append(f"\n--- Current Task (Iteration {iteration_num + 1}, Model: {target_model}) ---")

        if iteration_num == 0 and not current_questions:
             instruction = f"Generate {num_questions_to_generate} initial business-focused analytical questions based on the database summary"
             if user_query: instruction += f" and the user query/topic."
             else: instruction += "."
        else:
            instruction = (
                f"Based on the summary and the previously generated questions, generate {num_questions_to_generate} NEW, more complex, or follow-up analytical questions.\n"
                f"Explore different angles or deeper insights related to the existing questions or the initial query/summary.\n"
                f"Avoid generating questions that are semantically very similar to the previous ones."
                f"Make sure your response is in JSON format with the following keys only: questions"
                f"questions is a list of strings, each representing a question."
            )
        user_prompt_parts.append(instruction)
        user_prompt_parts.append(f"\nProvide ONLY the numbered list of {num_questions_to_generate} new questions as per the format instructions.")

        return system_prompt.strip(), "\n\n".join(user_prompt_parts)

    def generate_questions(
        self,
        db_summary: Union[str, Dict],
        user_query: Optional[str] = None
    ) -> GeneratedQuestions:
        """
        Generates analytical questions based on the configuration.

        Args:
            db_summary: Database summary (string or dict).
            user_query: Optional initial user query.

        Returns:
            A GeneratedQuestions object containing the results.
        """
        if not db_summary:
            raise ValueError("db_summary must be provided.")

        self.all_questions = [] # Reset questions for a new run
        total_iterations_run_per_model: Dict[str, int] = {model: 0 for model in self.config.llm_list}

        for i in range(self.config.max_iterations_per_model):
            logger.info(f"--- Starting Iteration {i + 1} ---")
            new_questions_this_iteration = []
            for model_name in self.config.llm_list:
                logger.info(f"Using model: {model_name} for iteration {i + 1}")

                # Build prompt with current context
                system_prompt, user_prompt = self._build_prompt(
                    db_summary=db_summary,
                    current_questions=self.all_questions,
                    target_model=model_name,
                    iteration_num=i,
                    num_questions_to_generate=self.config.num_questions_per_iteration_per_model,
                    user_query=user_query
                )

                # Call LLM
                if 'gpt' in model_name:
                    response_json = openai_call(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=model_name,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_output_tokens,
                        max_retries=3,
                        expected_keys=["questions"]
                    )
                else:
                    response_json = gemini_call(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        model=model_name,
                        temperature=self.config.llm_temperature,
                        max_tokens=self.config.llm_max_output_tokens,
                        thinking_budget=10000,
                        max_retries=3,
                        expected_keys=["questions"]
                    )

                # Parse the response
                questions_json = self._response_parser(
                    response=response_json["questions"], 
                    source_llm=model_name, 
                    iteration_level=i)
                
                new_questions_this_iteration.extend(questions_json)

                total_iterations_run_per_model[model_name] += 1

            

            # Add questions from this iteration to the main list
            self.all_questions.extend(new_questions_this_iteration)
            logger.info(f"Current questions: {self.all_questions}")
            logger.info(f"--- Finished Iteration {i + 1}. Total questions generated so far: {len(self.all_questions)} ---")

        final_output = GeneratedQuestions(
            questions=self.all_questions,
            models_used=list(self.config.llm_list),
            total_iterations_run_per_model=total_iterations_run_per_model,
            final_question_count=len(self.all_questions)
        )

        logger.info(f"Question generation complete. Generated {final_output.final_question_count} questions in total.")
        return final_output


# --- Main execution block (example) ---
if __name__ == "__main__":
    db_summary_path = "/Users/arshath/play/experiments/insights/database_summary.json"
    with open(db_summary_path, "r") as f:
        db_summary_content = json.load(f)

    # Create Config
    config = GeneratorConfig(
        llm_list=["gpt-4.1", "gemini-2.5-flash-preview-04-17"],
        max_iterations_per_model=4,
        num_questions_per_iteration_per_model=3,
        llm_temperature=0.2
    )

    # Create and run generator
    generator = IterativeBusinessQuestionGenerator(config)

    try:
        results = generator.generate_questions(
            db_summary=db_summary_content,
            user_query=None
        )

        # Save results
        try:
            with open("generated_questions.json", 'w', encoding='utf-8') as f:
                f.write(results.model_dump_json(indent=2))
            logger.info(f"Successfully generated {results.final_question_count} questions and saved to generated_questions.json")
        except Exception as e:
            logger.error(f"Error saving results to generated_questions.json: {e}")
            # Optionally print results if saving failed
            print("\n--- Generated Questions ---")
            print(results.model_dump_json(indent=2))

    except ValueError as ve:
         logger.error(f"Configuration or input error: {ve}")
         exit(1)
    except Exception as e:
         logger.exception(f"An unexpected error occurred during question generation: {e}")
         exit(1)