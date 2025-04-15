import json
import logging
import statistics
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

from insights.llm import call_openai_api
from insights.utils import setup_logging
from insights.config import OPENAI_API_KEY, DB_CONFIG
logger = setup_logging()

class InsightTier(str, Enum):
    """Tiers representing the depth of an insight."""
    OBSERVATION = "observation"  # Simple fact from the data
    COMPARISON = "comparison"    # Comparison between two or more data points
    SIGNIFICANCE = "significance"  # Statistical significance assessment
    PATTERN = "pattern"          # Identified trend or pattern
    CONTRIBUTION = "contribution"  # Causal analysis or contribution assessment

class Insight(BaseModel):
    """Model for a structured insight derived from data."""
    insight_id: str
    question_id: str  # ID of the question that generated this insight
    insight_tier: InsightTier
    headline: str  # Concise summary
    description: str  # Detailed explanation
    supporting_data: Dict[str, Any] = Field(default_factory=dict)  # Key evidence from results
    comparison_point: Optional[Dict[str, Any]] = None  # Benchmark or historical data
    significance_metric: Optional[Dict[str, Any]] = None  # Statistical context
    confidence_score: float = Field(ge=0.0, le=1.0)  # Confidence in the insight
    visualization_suggestion: Optional[str] = None  # Suggested visualization type

class InsightResult(BaseModel):
    """Model for the complete insight analysis result."""
    query: str  # Original SQL query
    question: str  # Original question
    question_id: str  # ID of the question
    insights: List[Insight]  # Generated insights
    total_insights: int  # Total number of insights generated
    tier_distribution: Dict[str, int] = Field(default_factory=dict)  # Distribution by tier

class InsightResults(BaseModel):
    """Model for the complete insight analysis result."""
    insights: List[Insight]

class InsightAgent:
    """
    Agent for generating insights from SQL query results.
    
    This agent analyzes data returned from SQL queries and generates
    structured insights at various levels of depth.
    """
    
    def __init__(self, 
                 openai_model: str = "gpt-4o-mini",
                 min_confidence_threshold: float = 0.7):
        """
        Initialize the Insight Agent.
        
        Args:
            openai_model: OpenAI model to use for insight generation
            min_confidence_threshold: Minimum confidence score for insights to be included
        """
        self.openai_model = openai_model
        self.min_confidence_threshold = min_confidence_threshold
        
    def _generate_insight_id(self, question_id: str, index: int) -> str:
        """
        Generate a unique insight ID.
        
        Args:
            question_id: The question ID that generated this insight
            index: The insight index for this question
            
        Returns:
            A formatted insight ID (e.g., Q001-I001)
        """
        return f"{question_id}-I{str(index+1).zfill(3)}"
    
    def _analyze_data_statistics(self, 
                               data: List[Dict[str, Any]],
                               column_names: List[str]) -> Dict[str, Any]:
        """
        Calculate basic statistics on the data.
        
        Args:
            data: The data to analyze
            column_names: Names of columns in the data
            
        Returns:
            Dictionary of statistical information
        """
        stats = {}
        
        if not data:
            return stats
            
        # Analyze each column
        for col in column_names:
            # Extract column values (skipping nulls/None)
            values = [row.get(col) for row in data if col in row and row[col] is not None]
            
            # Skip empty columns
            if not values:
                continue
                
            col_stats = {}
            
            # Check if values are numeric
            try:
                numeric_values = [float(v) for v in values if v != ""]
                if numeric_values:
                    col_stats["min"] = min(numeric_values)
                    col_stats["max"] = max(numeric_values)
                    col_stats["mean"] = statistics.mean(numeric_values)
                    col_stats["median"] = statistics.median(numeric_values)
                    if len(numeric_values) > 1:
                        col_stats["std_dev"] = statistics.stdev(numeric_values)
            except (ValueError, TypeError):
                # Not numeric, treat as categorical
                value_counts = {}
                for v in values:
                    if v in value_counts:
                        value_counts[v] += 1
                    else:
                        value_counts[v] = 1
                        
                # Get top categories
                sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                col_stats["value_counts"] = {k: v for k, v in sorted_counts[:10]}  # Top 10 categories
                col_stats["unique_count"] = len(value_counts)
                
            # Add to overall stats
            stats[col] = col_stats
            
        return stats
    
    def _detect_time_series(self, 
                           data: List[Dict[str, Any]],
                           column_names: List[str]) -> Optional[str]:
        """
        Detect if data has a time series column.
        
        Args:
            data: The data to analyze
            column_names: Names of columns in the data
            
        Returns:
            Name of time series column if found, None otherwise
        """
        if not data:
            return None
            
        # Look for date/time columns
        for col in column_names:
            # Check first few values to see if they could be dates
            sample_values = [row.get(col) for row in data[:5] if col in row and row[col] is not None]
            
            if not sample_values:
                continue
                
            # Try to parse as dates
            try:
                # Try different date formats
                date_formats = [
                    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"
                ]
                
                for fmt in date_formats:
                    try:
                        # Try to parse the first value
                        datetime.strptime(str(sample_values[0]), fmt)
                        return col  # Success - this is likely a date column
                    except ValueError:
                        continue
            except:
                continue
                
        return None
    
    def _extract_context_from_db_summary(self, 
                                       db_summary: Dict[str, Any],
                                       tables: List[str]) -> str:
        """
        Extract relevant context from the database summary for the specified tables.
        
        Args:
            db_summary: The database summary
            tables: Tables mentioned in the query
            
        Returns:
            Relevant context as a string
        """
        if not db_summary or not tables:
            return ""
            
        context = []
        
        # Get natural language summary
        nl_summary = db_summary.get("natural_language_summary", "")
        if nl_summary:
            context.append(nl_summary)
            
        # Extract information about the specified tables
        tech_summary = db_summary.get("technical_summary", {})
        db_tables = tech_summary.get("tables", [])
        
        for table in db_tables:
            table_name = table.get("name", "")
            if table_name in tables:
                # Add table information
                row_count = table.get("row_count", 0)
                context.append(f"Table {table_name} has {row_count} rows.")
                
                # Add information about time data if available
                if table.get("has_time_data", False) and table.get("time_range"):
                    time_info = []
                    for col, range_info in table.get("time_range", {}).items():
                        time_info.append(f"Column {col} has date range from {range_info.get('min')} to {range_info.get('max')}.")
                    context.extend(time_info)
                    
                # Add column information for columns with few unique values
                for column in table.get("columns", []):
                    if "distribution" in column and column.get("distinct_count", 0) <= 20:
                        col_name = column.get("name", "")
                        dist = column.get("distribution", {})
                        context.append(f"Column {col_name} in table {table_name} has these values: {dist}")
        
        return "\n".join(context)
    
    def generate_insights(self, 
                         query_result: Dict[str, Any],
                         user_query: str,
                         db_summary: Dict[str, Any]) -> InsightResult:
        """
        Generate insights from SQL query results.
        
        Args:
            query_result: Result from the Text2SQL agent
            user_query: Original user query
            db_summary: Summary of the database
            
        Returns:
            Structured insights about the data
        """
        # Extract necessary information
        sql_query = query_result.get("sql", "")
        question = query_result.get("question", "")
        question_id = query_result.get("question_id", "Q000")
        execution_result = query_result.get("execution_result", {})
        
        # Check if the query was successful
        if not execution_result.get("success", False):
            logger.error(f"Cannot generate insights for failed query: {execution_result.get('error_message')}")
            return InsightResult(
                query=sql_query,
                question=question,
                question_id=question_id,
                insights=[],
                total_insights=0
            )
            
        # Extract data
        data = execution_result.get("data", [])
        if not data:
            logger.warning(f"No data returned from query: {sql_query}")
            return InsightResult(
                query=sql_query,
                question=question,
                question_id=question_id,
                insights=[],
                total_insights=0
            )
            
        # Extract column names
        column_names = execution_result.get("column_names", [])
        
        # Calculate statistics on the data
        data_stats = self._analyze_data_statistics(data, column_names)
        
        # Detect if there's a time series column
        time_column = self._detect_time_series(data, column_names)
        
        # Extract tables from the query
        tables = []
        
        # Simple parsing to extract table names from the SQL query
        sql_lower = sql_query.lower()
        from_pos = sql_lower.find("from ")
        if from_pos != -1:
            where_pos = sql_lower.find("where", from_pos)
            group_pos = sql_lower.find("group by", from_pos)
            order_pos = sql_lower.find("order by", from_pos)
            having_pos = sql_lower.find("having", from_pos)
            
            end_pos = min([pos for pos in [where_pos, group_pos, order_pos, having_pos] if pos != -1], default=len(sql_lower))
            
            tables_str = sql_lower[from_pos + 5:end_pos].strip()
            tables = [t.strip().split(" ")[-1] for t in tables_str.split(",")]
            
            # Handle joins
            join_keywords = ["join", "inner join", "left join", "right join", "full join"]
            for keyword in join_keywords:
                pos = 0
                while True:
                    pos = sql_lower.find(f"{keyword} ", pos)
                    if pos == -1:
                        break
                    on_pos = sql_lower.find(" on ", pos)
                    if on_pos == -1:
                        break
                    table_name = sql_lower[pos + len(keyword) + 1:on_pos].strip()
                    tables.append(table_name.split(" ")[-1])
                    pos = on_pos
        
        # Extract context from database summary
        db_context = self._extract_context_from_db_summary(db_summary, tables)
        
        # Prepare data for the LLM
        # Limit to a reasonable number of rows to avoid token limits
        max_rows = 100
        data_sample = data[:max_rows]
        
        # Prepare prompt for the LLM
        system_prompt = """
        You are an expert data analyst who specializes in finding insightful patterns in SQL query results.
        
        Your task is to analyze the data provided and generate multiple insights with different levels of depth:
        
        1. OBSERVATION: Simple factual observations directly from the data
        2. COMPARISON: Comparisons between different data points, segments, or categories
        3. SIGNIFICANCE: Assessments of statistical significance or importance
        4. PATTERN: Identification of trends, patterns, or correlations
        5. CONTRIBUTION: Analysis of potential causal factors or contributions
        
        For each insight, provide:
        - An insight tier (one of the five levels above)
        - A concise headline summarizing the insight
        - A detailed explanation with context
        - Supporting data (specific numbers or facts from the data)
        - Comparison points (if applicable)
        - Statistical metrics (if applicable)
        - A confidence score (0.0-1.0) indicating your confidence in the insight
        - A suggestion for how to visualize this insight (if applicable)
        
        Your insights should be data-driven, specific, and actionable. They should answer the original question and provide additional value beyond simple observations.
        """
        
        user_prompt = f"""
        ORIGINAL USER QUERY: "{user_query}"
        
        QUESTION: "{question}"
        
        SQL QUERY: {sql_query}
        
        QUERY RESULTS (first {len(data_sample)} rows):
        {json.dumps(data_sample, indent=2)}
        
        COLUMN STATISTICS:
        {json.dumps(data_stats, indent=2)}
        
        CONTEXT FROM DATABASE:
        {db_context}
        
        TIME SERIES COLUMN (if identified): {time_column if time_column else "None"}
        
        Based on the above information, generate 3-5 insights at different levels of depth (observation, comparison, significance, pattern, contribution).
        
        Format each insight as a JSON object with the following structure:
        {{
            "insight_id": "I001",  // Will be assigned later
            "insight_tier": "one of: observation, comparison, significance, pattern, contribution",
            "headline": "Concise summary of the insight",
            "description": "Detailed explanation with context",
            "supporting_data": {{ "key metrics or evidence": "values" }},
            "comparison_point": {{ "benchmark or comparison": "values" }} or null if not applicable,
            "significance_metric": {{ "statistical context": "values" }} or null if not applicable,
            "confidence_score": 0.85,  // Between 0.0 and 1.0
            "visualization_suggestion": "Suggested chart type" or null if not applicable
        }}
        
        Return an array of these insight objects. The key of the json object is "insights".
        """
        
        # Call LLM to generate insights
        logger.info(f"Generating insights for question: {question}")
        insights_data = call_openai_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=InsightResults,
            model=self.openai_model,
            temperature=0.5
        )
        
        if not insights_data:
            logger.error(f"Failed to generate insights for question: {question}")
            return InsightResult(
                query=sql_query,
                question=question,
                question_id=question_id,
                insights=[],
                total_insights=0
            )
        
        # Add question_id and generate insight_id for each insight
        for i, insight in enumerate(insights_data.insights):
            insight.question_id = question_id
            insight.insight_id = self._generate_insight_id(question_id, i)
        
        # Filter low-confidence insights
        filtered_insights = [i for i in insights_data.insights if i.confidence_score >= self.min_confidence_threshold]
        
        # Calculate tier distribution
        tier_distribution = {}
        for insight in filtered_insights:
            tier = insight.insight_tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
        
        return InsightResult(
            query=sql_query,
            question=question,
            question_id=question_id,
            insights=filtered_insights,
            total_insights=len(filtered_insights),
            tier_distribution=tier_distribution
        )
    
    def process_query_results(self, 
                            query_results: List[Dict[str, Any]],
                            user_query: str,
                            db_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multiple query results and generate insights for each.
        
        Args:
            query_results: List of results from the Text2SQL agent
            user_query: Original user query
            db_summary: Summary of the database
            
        Returns:
            Dictionary with insights for each query result
        """
        all_insights = []
        
        for query_result in query_results:
            # Check if query was successful
            if "execution_result" not in query_result or not query_result["execution_result"].get("success", False):
                logger.warning(f"Skipping failed query result: {query_result.get('question', 'Unknown question')}")
                continue
                
            # Generate insights for this query result
            insights = self.generate_insights(query_result, user_query, db_summary)
            
            # Add to overall results
            all_insights.append(insights.model_dump())
        
        return {
            "user_query": user_query,
            "total_results": len(all_insights),
            "insights": all_insights
        }

def main(query_results_path: str, db_summary_path: str, user_query: str, output_path: str = "insights.json"):
    """
    Entry point function to run the Insight Agent.
    
    Args:
        query_results_path: Path to query results JSON file
        db_summary_path: Path to database summary JSON file
        user_query: Original user query
        output_path: Path to save insights JSON file
    """
    # Load query results
    with open(query_results_path, 'r') as f:
        query_results = json.load(f)
    
    # Load database summary
    with open(db_summary_path, 'r') as f:
        db_summary = json.load(f)
    
    # Create insight agent
    agent = InsightAgent()
    
    # Process query results
    insights = agent.process_query_results(query_results, user_query, db_summary)
    
    # Save insights
    with open(output_path, 'w') as f:
        json.dump(insights, f, indent=2)
    
    logger.info(f"Generated insights for {len(insights['insights'])} query results. Saved to {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python insight_agent.py <query_results_path> <db_summary_path> <user_query> ")
        sys.exit(1)
    
    query_results_path = sys.argv[1]
    db_summary_path = sys.argv[2]
    user_query = sys.argv[3]
    
    main(query_results_path, db_summary_path, user_query)