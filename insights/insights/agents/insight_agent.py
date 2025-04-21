import json
import uuid
import pandas as pd
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError
import concurrent.futures

# --- Local Imports ---
from insights.agents.db_summary_agent import DatabaseSummary
from insights.agents.text2sql_agent import ProcessedQuestionResult
from insights.utils import setup_logging
from insights.llm import LLM
from insights.config import (
    INSIGHT_AGENT_LLM_PROVIDER, MAX_CONCURRENT_WORKERS
)

logger = setup_logging()


# --- Pydantic Models for Insights ---
class InsightTier(str, Enum):
    """Categorizes the primary nature or depth of the insight."""
    OBSERVATION = "Observation"       # Stating a fact derived from the data (e.g., total sales).
    COMPARISON = "Comparison"         # Comparing values within the result set or against summary stats.
    TREND = "Trend"                   # Identifying a pattern over time within the result set.
    ANOMALY = "Anomaly"               # Highlighting significant deviations within the result set or vs. summary stats.
    CONTRIBUTION = "Contribution"     # Identifying key drivers/segments responsible for an outcome in the results.
    RELATIONSHIP = "Relationship"     # Describing a correlation or link found *within* the result data (e.g., between two metrics).
    DATA_QUALITY = "Data Quality"     # Highlighting potential issues found in the result data (e.g., unexpected NULLs, outliers suggesting errors).
    EXECUTION_INFO = "Execution Info" # Insight about the query execution itself (e.g., failure, long runtime, empty result set).

class StructuredInsight(BaseModel):
    """Represents a single, structured insight derived from query results."""
    insight_id: str = Field(default_factory=lambda: f"INS-{uuid.uuid4().hex[:12]}", description="Unique identifier for the insight.")
    question_id: str = Field(..., description="ID of the AnalysisQuestion this insight addresses.")
    question_text: str = Field(..., description="The original question text for context.")

    headline: str = Field(..., min_length=5, max_length=150, description="A concise, impactful headline summarizing the core insight.")
    description: str = Field(..., min_length=10, description="Detailed explanation of the insight, its context, potential implications (the 'so what?').")
    tier: InsightTier = Field(..., description="The category classifying the type of analysis/finding.")

    # --- Supporting Evidence & Context ---
    supporting_metrics: Dict[str, Any] = Field(default_factory=dict, description="Key aggregated data points/metrics from results supporting the insight (e.g., {'Region A Sales': 1200, 'Region B Sales': 1000, '% Change': 20.0}).")
    supporting_examples: Optional[List[Dict[str, Any]]] = Field(None, max_items=5, description="Optional: 1-3 sample rows from results illustrating the point, if metrics aren't sufficient.")

    # Specific fields for certain tiers (Optional, populated based on tier)
    comparison_details: Optional[str] = Field(None, description="Describes the baseline for COMPARISON or ANOMALY tiers (e.g., 'Region B', 'Previous Month', 'Overall Average').")
    trend_pattern: Optional[str] = Field(None, description="Describes the pattern for TREND tier (e.g., 'Consistent Increase', 'Seasonal Peak').")
    anomaly_description: Optional[str] = Field(None, description="Specific description of the deviation for ANOMALY tier.")
    contribution_details: Optional[str] = Field(None, description="Details for CONTRIBUTION tier (e.g., 'Top 3 Categories accounted for 85%').")

    # --- Evaluation & Next Steps ---
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Estimated relevance to the original user query/question (0.0-1.0).")
    significance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Estimated impact, surprise factor, or magnitude (0.0-1.0).")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Estimated confidence in the finding based on data quality/ambiguity (0.0-1.0).")

    potential_actions: List[str] = Field(default_factory=list, description="Suggested business actions or decisions based on the insight.")
    further_investigation_q: List[str] = Field(default_factory=list, description="Suggested follow-up analytical questions.")

    # --- Source Provenance ---
    source_sql: str = Field(..., description="The SQL query that generated the data for this insight.")
    data_row_count: Optional[int] = Field(None, description="Number of rows in the result set used for this insight.")
    data_column_names: Optional[List[str]] = Field(None, description="Columns present in the result set.")
    execution_time: Optional[float] = Field(None, description="Execution time of the source SQL query.")
    error_info: Optional[str] = Field(None, description="Error message if the insight relates to a data retrieval/processing failure.")

    @field_validator('headline', 'description')
    @classmethod
    def check_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Headline and description cannot be empty or just whitespace.")
        return v.strip()

class LLMResponse(BaseModel):
    insights: List[StructuredInsight]

# --- Insight Agent Class ---

class InsightAgent:
    """
    Analyzes results from a single query execution (ProcessedQuestionResult)
    and generates structured insights using an LLM.
    """
    def __init__(self):
        """
        Initializes the Insight Agent.
        """
        self.llm = LLM.create_client(INSIGHT_AGENT_LLM_PROVIDER)
        logger.info("InsightAgent initialized.")

    def generate_insights(self,
                          processed_questions: List[ProcessedQuestionResult],
                          db_summary: DatabaseSummary,
                          original_user_query: Optional[str] = None) -> List[StructuredInsight]:
        """
        Generates a list of structured insights for a given ProcessedQuestionResult.
        """

        insights = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
            futures = [executor.submit(self.generate_insights_per_question, processed_question, db_summary, original_user_query) for processed_question in processed_questions]
            for future in concurrent.futures.as_completed(futures):
                insights.extend(future.result())
        return insights

    def generate_insights_per_question(self,
                          processed_question: ProcessedQuestionResult,
                          db_summary: DatabaseSummary,
                          original_user_query: Optional[str] = None) -> List[StructuredInsight]:
        """
        Generates a list of structured insights for a given ProcessedQuestionResult.

        Args:
            processed_question: The result object from the Text2SQLExecuteAgent.
            db_summary: The summary of the database for context.
            original_user_query: The initial high-level query from the user.

        Returns:
            A list of StructuredInsight objects. Returns an empty list if no
            insights could be generated or if input indicates failure/no data
            and error insights are generated instead.
        """


        qid = processed_question.question_id

        # --- 1. Handle Execution Errors or Processing Errors ---
        if not processed_question.execution_success or processed_question.processing_error:
            error_msg = processed_question.execution_error_message or processed_question.processing_error or "Unknown error"
            headline = "Query Execution or Processing Failed"
            description = f"Could not retrieve or process data for question ID '{qid}': '{processed_question.question_text}'. Error: {error_msg}"
            if processed_question.generated_sql:
                 description += f"\nAttempted SQL: {processed_question.generated_sql}"

            error_insight = StructuredInsight(
                question_id=qid,
                question_text=processed_question.question_text,
                headline=headline,
                description=description,
                tier=InsightTier.EXECUTION_INFO,
                source_sql=processed_question.generated_sql or "SQL not generated or processing failed earlier",
                error_info=error_msg,
                data_row_count=0,
                data_column_names=processed_question.execution_column_names or [],
                execution_time=processed_question.execution_time,
                relevance_score=1.0, # High relevance as it blocks analysis
                significance_score=0.5 # Depends on question criticality
            )
            logger.warning(f"Generating EXECUTION_INFO insight for failed question ID: {qid}")
            return [error_insight]

        # --- 2. Handle Empty Result Sets ---
        if not processed_question.execution_data:
             headline = "Query Returned No Data"
             description = f"The query for question ID '{qid}': '{processed_question.question_text}' executed successfully but returned zero matching records."
             empty_insight = StructuredInsight(
                 question_id=qid,
                 question_text=processed_question.question_text,
                 headline=headline,
                 description=description,
                 tier=InsightTier.OBSERVATION, # Could also be EXECUTION_INFO
                 source_sql=processed_question.generated_sql,
                 data_row_count=0,
                 data_column_names=processed_question.execution_column_names or [],
                 execution_time=processed_question.execution_time,
                 relevance_score=0.8, # Relevant that no data exists
                 significance_score=0.3 # Significance depends on whether data was expected
             )
             logger.info(f"Generating OBSERVATION insight for empty result set for question ID: {qid}")
             return [empty_insight]

        prompt = self._build_insight_prompt(
            processed_question,
            db_summary,
            original_user_query
        )
        llm_response = self.llm.generate(
            system_prompt="You are an expert data analyst. Your task is to analyze the provided query results in context and generate structured, actionable insights in JSON format according to the provided schema. Focus on comparisons, trends, anomalies, and significance. Provide ONLY the JSON object containing an 'insights' list.",
            user_prompt=prompt,
            response_model=LLMResponse
        )
        return llm_response.insights

    def _build_insight_prompt(self,
                              processed_question: ProcessedQuestionResult,
                              db_summary: DatabaseSummary,
                              original_user_query: Optional[str]) -> str:
        """Constructs the detailed prompt for the LLM to generate insights."""

        # --- Summarize Execution Data ---
        data_summary_str = self._summarize_execution_data(
            processed_question.execution_data or [], # Handle potential None
            processed_question.execution_column_names or []
        )

        # --- Prepare Database Context ---
        # Keep context concise, focus on potentially relevant parts if possible
        db_nl_summary = "Not available"
        if db_summary and hasattr(db_summary, 'natural_language_summary') and db_summary.natural_language_summary:
            db_nl_summary = db_summary.natural_language_summary

        db_context_str = f"General Database Context:\n{db_nl_summary}\n"

        # --- Define Tiers for Prompt ---
        allowed_tiers = [t.value for t in InsightTier if t not in [InsightTier.EXECUTION_INFO]]
        tier_enum_str = ", ".join(allowed_tiers)

        # --- Construct Prompt ---
        prompt = f"""
Analyze the following data query results to generate insightful findings.

**1. Analysis Task Context:**
   - Original User Query (if provided): "{original_user_query or 'Not provided'}"
   - Specific Question Asked: "{processed_question.question_text}"
   - SQL Query Used: ```sql
{processed_question.generated_sql}
```

**2. Data Query Results Summary:**
   - Row Count: {processed_question.execution_row_count}
   - Columns: {", ".join(processed_question.execution_column_names or [])}
   - Data Summary / Snippet:
     ```
     {data_summary_str}
     ```

**3. Broader Database Context:**
   {db_context_str}

**4. Your Task:**
   - Analyze the 'Data Query Results Summary' considering the 'Specific Question Asked' and 'Broader Database Context'.
   - Identify 1-3 distinct, meaningful insights based *only* on the provided data and context. Do not assume external knowledge or data not present.
   - Focus on findings like: Comparisons between groups/segments, simple Trends (if time data exists in results), significant Anomalies/deviations, Contributions of segments to totals, or interesting Relationships between metrics.
   - For each insight found, provide:
     - A concise `headline` (5-150 chars).
     - A detailed `description` explaining the finding and its importance/implications (min 10 chars).
     - An appropriate `tier` (must be one of: {tier_enum_str}).
     - A `supporting_metrics` dictionary with key calculated values supporting the headline (e.g., averages, totals, percentage changes).
     - Optionally, `comparison_details`, `trend_pattern`, `anomaly_description`, or `contribution_details` specific to the insight tier.
     - Estimate `relevance_score`, `significance_score`, and `confidence_score` (0.0-1.0, null if not applicable).
     - Suggest 1-2 `potential_actions` (brief business actions) and 1-2 `further_investigation_q` (follow-up questions).
   - **Format your entire response strictly as a single JSON object with a single key "insights". The value of "insights" must be a list of JSON objects, each conforming to the StructuredInsight schema provided below.**
"""
        return prompt.strip()


    def _summarize_execution_data(self, data: List[Dict[str, Any]], columns: List[str]) -> str:
        """Creates a concise text summary of the execution data for the LLM prompt."""
        if not data:
            return "No data returned."

        num_rows = len(data)
        summary_lines = []
        max_summary_chars = 1500 # Limit summary size to avoid huge prompts

        try:
            # Use Pandas for quick stats if available and data is suitable
            df = pd.DataFrame(data)
            current_len = 0

            # Numeric Stats
            numeric_cols = df.select_dtypes(include='number').columns
            if not numeric_cols.empty:
                 stats = df[numeric_cols].describe().to_string()
                 if len(stats) < max_summary_chars - current_len:
                     summary_lines.append("Numeric Column Statistics:\n" + stats)
                     current_len += len(stats) + 28 # Approx length of header

            # Categorical Stats (Value Counts)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if current_len < max_summary_chars and not categorical_cols.empty:
                summary_lines.append("\nCategorical Column Value Counts (Top 5):")
                current_len += 40
                for col in categorical_cols:
                    counts_header = f"\n-- Column '{col}' --\n"
                    counts = df[col].value_counts().head(5).to_string()
                    if len(counts_header) + len(counts) < max_summary_chars - current_len:
                        summary_lines.append(counts_header + counts)
                        current_len += len(counts_header) + len(counts)
                    else:
                        summary_lines.append(f"\n-- Column '{col}' (Skipped due to length) --")
                        break # Stop adding categorical summaries if too long

            # Fallback/Supplement: Row Samples if stats are missing or short
            if not summary_lines or current_len < 500: # Add samples if summary is very short
                 sample_count = min(num_rows, 3) # Show fewer rows if stats are present
                 header = f"\n\nSample Rows (First {sample_count} of {num_rows}):"
                 summary_lines.append(header)
                 current_len += len(header)
                 for i in range(sample_count):
                     row_dict = data[i]
                     # Simple string conversion, truncate long values
                     row_str = ", ".join([f"{k}: '{str(v)[:30]}...'" if isinstance(v, str) and len(str(v)) > 30 else f"{k}: {v}" for k, v in row_dict.items()])
                     line = f"- {row_str}"
                     if len(line) < max_summary_chars - current_len:
                         summary_lines.append(line)
                         current_len += len(line)
                     else:
                         summary_lines.append("- (Row skipped due to length)")
                         break
                 if num_rows > sample_count and current_len < max_summary_chars:
                     summary_lines.append(f"... ({num_rows - sample_count} more rows total)")


        except ImportError:
            logger.warning("Pandas not installed. Using basic row sampling for data summary.")
            # Basic row sampling without pandas
            sample_count = min(num_rows, 5)
            summary_lines.append(f"Sample Rows (First {sample_count} of {num_rows}):")
            for i in range(sample_count):
                 row_dict = data[i]
                 row_str = ", ".join([f"{k}: '{str(v)[:30]}...'" if isinstance(v, str) and len(str(v)) > 30 else f"{k}: {v}" for k, v in row_dict.items()])
                 summary_lines.append(f"- {row_str}")
            if num_rows > sample_count:
                summary_lines.append(f"... ({num_rows - sample_count} more rows total)")

        except Exception as e:
             logger.error(f"Error during data summarization: {e}", exc_info=True)
             return f"Error summarizing data: {e}. First row: {str(data[0]) if data else 'N/A'}"

        return "\n".join(summary_lines)[:max_summary_chars] # Ensure final length limit
