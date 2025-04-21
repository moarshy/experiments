import logging
import datetime
import json
from typing import List, Optional, Dict, Any

from insights.agents.insight_agent import StructuredInsight, InsightTier
from insights.agents.db_summary_agent import DatabaseSummary
from insights.llm import LLM
from insights.utils import setup_logging

logger = setup_logging(__name__)

# --- Default Configuration ---
DEFAULT_TARGET_AUDIENCE = "executive"
DEFAULT_REPORT_FORMAT = "markdown" # Currently only markdown supported via LLM prompt
DEFAULT_INSIGHTS_LIMIT_FOR_PROMPT = 15 # Max insights to include directly in prompt
DEFAULT_REPORT_STRUCTURE = """
# Report Title (Generate a suitable title based on the query/database)

*Generated on: {current_date}*

## Executive Summary
(Provide a brief overview of the analysis purpose and highlight 2-3 most critical findings based on the insights provided.)

## Key Findings
(Detail the most important insights from the provided list. For each, include the headline, a clear description of the finding and its implication, and key supporting metrics if available. Use Markdown formatting.)

## Recommendations / Potential Actions
(Summarize the actionable recommendations suggested by the insights.)

## Areas for Further Investigation
(List the key questions raised for further analysis based on the insights.)

## Conclusion
(Provide a brief concluding statement.)
"""

# --- Report Generating Agent (v2.0 - LLM Powered) ---

class ReportGeneratingAgent:
    """
    Generates a human-readable report in Markdown by prompting an LLM
    with consolidated insights and context.
    """

    def __init__(self,
                 llm_provider: str = "openai",
                 llm_model_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Report Generating Agent.

        Args:
            llm_provider: The LLM provider ('openai' or 'gemini').
            llm_model_config: Dictionary with model name and other LLM client args.
        """
        self.llm_client = LLM.create_client(provider=llm_provider, **(llm_model_config or {}))
        if self.llm_client is None:
             raise ValueError(f"Could not create LLM client for provider: {llm_provider}")
        logger.info(f"ReportGeneratingAgent initialized with LLM provider: {llm_provider}")

    def generate_report(self,
                        consolidated_insights: List[StructuredInsight],
                        original_user_query: Optional[str] = None,
                        database_summary: Optional[DatabaseSummary] = None,
                        target_audience: str = DEFAULT_TARGET_AUDIENCE,
                        report_format: str = DEFAULT_REPORT_FORMAT,
                        report_structure_prompt: str = DEFAULT_REPORT_STRUCTURE,
                        insights_limit: int = DEFAULT_INSIGHTS_LIMIT_FOR_PROMPT
                        ) -> str:
        """
        Generates the report content string using an LLM.

        Args:
            consolidated_insights: The final list of prioritized insights.
            original_user_query: The initial user query for context.
            database_summary: Summary of the database for context.
            target_audience: Specifies the audience ('executive', 'technical', etc.).
            report_format: Specifies the output format (currently only 'markdown' supported).
            report_structure_prompt: A string outlining the desired report sections.
            insights_limit: Max number of insights to include in the LLM prompt.

        Returns:
            The LLM-generated report as a Markdown string.
        """
        if report_format.lower() != 'markdown':
            logger.error(f"Report format '{report_format}' not supported in this LLM-based version. Only 'markdown' is available.")
            return f"# Unsupported Report Format\n\nRequested format: {report_format}. Only Markdown is currently supported."

        if not consolidated_insights:
            logger.warning("No insights provided to generate report. Returning empty report.")
            # Provide a basic structure even for no insights
            report_title = "Data Insights Report"
            if original_user_query: report_title += f": Analysis for '{original_user_query}'"
            return f"# {report_title}\n*Generated on: {datetime.date.today().isoformat()}*\n\n## Executive Summary\nNo insights were generated or provided for reporting."

        logger.info(f"Generating LLM-based {report_format} report for {target_audience} audience from {len(consolidated_insights)} insights (using top {insights_limit})...")

        # 1. Prepare data and build the prompt
        try:
            prompt = self._build_report_prompt(
                insights=consolidated_insights[:insights_limit], # Limit insights for prompt
                original_query=original_user_query,
                db_summary=database_summary,
                structure_prompt=report_structure_prompt,
                audience=target_audience
            )
        except Exception as e:
            logger.error(f"Failed to build report prompt: {e}", exc_info=True)
            return f"# Report Generation Error\n\nFailed to construct the prompt for the language model. Error: {e}"

        # 2. Call the LLM to generate the report
        try:
            # Use the generate method expecting string output (the markdown report)
            report_markdown = self.llm_client.generate(
                user_prompt=prompt,
                system_prompt=f"You are a helpful assistant tasked with writing a clear and concise data analysis report in Markdown format for a '{target_audience}' audience. Use the provided insights and structure.",
                response_model=None, # Expecting markdown string, not structured JSON
                temperature=0.3, # Lower temperature for more factual reporting
                # max_tokens might be needed depending on report length
            )
            if not report_markdown or not report_markdown.strip():
                 raise ValueError("LLM returned an empty report.")

            logger.info("Successfully generated report using LLM.")
            return report_markdown

        except Exception as e:
            logger.error(f"LLM call failed during report generation: {e}", exc_info=True)
            return f"# Report Generation Error\n\nFailed to generate the report using the language model. Error: {e}"


    def _build_report_prompt(self,
                             insights: List[StructuredInsight],
                             original_query: Optional[str],
                             db_summary: Optional[DatabaseSummary],
                             structure_prompt: str,
                             audience: str) -> str:
        """Constructs the detailed prompt for the LLM to generate the report."""

        # --- Serialize Insights for Prompt ---
        # Convert insights to a readable string format (e.g., JSON or formatted list)
        # Include key fields relevant for reporting.
        insights_str_list = []
        for i, insight in enumerate(insights):
            # Select key fields to include in the prompt context
            insight_context = {
                "id": insight.insight_id,
                "headline": insight.headline,
                "description": insight.description,
                "tier": str(getattr(insight.tier, 'value', insight.tier)), # Get enum value if possible
                "supporting_metrics": insight.supporting_metrics,
                "potential_actions": insight.potential_actions,
                "further_investigation_q": insight.further_investigation_q,
                "relevance": insight.relevance_score,
                "significance": insight.significance_score,
            }
            # Simple formatted string representation
            insights_str_list.append(f"--- Insight {i+1} ---\n{json.dumps(insight_context, indent=2)}")

        insights_input_str = "\n\n".join(insights_str_list)
        if not insights_input_str:
            insights_input_str = "No specific insights were provided."

        # --- Prepare Context ---
        db_context = "Database: Not specified."
        if db_summary and db_summary.database_name:
            db_context = f"Database Analyzed: '{db_summary.database_name}'"
            if db_summary.natural_language_summary:
                 db_context += f"\nDatabase Summary: {db_summary.natural_language_summary}"

        query_context = f"Analysis Focus (Original Query): {original_query}" if original_query else "Analysis Focus: General data exploration."

        # --- Prepare Structure Prompt ---
        # Replace placeholders like {current_date}
        structure_prompt = structure_prompt.format(current_date=datetime.date.today().isoformat())

        # --- Construct Final Prompt ---
        prompt = f"""
Your task is to generate a data analysis report in **Markdown format**.

**Target Audience:** {audience}
(Adjust tone, level of detail, and technical jargon accordingly. E.g., for 'executive', be concise, focus on impact and actions. For 'technical', include more detail.)

**Context:**
{query_context}
{db_context}

**Provided Insights (Prioritized List):**
(Use these insights as the primary source material for the report content. Synthesize, summarize, and integrate them into the report structure below. Focus on the most relevant and significant ones based on the context and scores provided within the insights.)
```json
[
{insights_input_str}
]
```

**Required Report Structure:**
(Follow this structure precisely. Populate each section using the provided insights and context. Generate appropriate narrative text.)
```markdown
{structure_prompt}
```

**Instructions:**
- Write the entire report based *only* on the provided insights and context. Do not add external information.
- Ensure the report flows logically and tells a coherent story based on the findings.
- Use Markdown formatting effectively (headings, lists, bolding).
- Adhere strictly to the requested structure.
- Populate all sections outlined in the structure. If no relevant information exists for a section based on the insights (e.g., no 'potential_actions' were suggested), state that clearly (e.g., "No specific actions were identified based on these findings.").

Generate the complete Markdown report now.
"""
        return prompt.strip()