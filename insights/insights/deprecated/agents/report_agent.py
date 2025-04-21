import json
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import markdown
import os
from datetime import datetime

from insights.llm import call_openai_api
from insights.utils import setup_logging
from insights.agents.insight_agent import InsightTier
from insights.agents.insights_consolidation_agent import Insight, SynthesizedInsight, ConsolidatedInsights

logger = setup_logging()

class ReportFormat(str, Enum):
    """Formats for the generated report."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"

class ReportSection(str, Enum):
    """Sections in the generated report."""
    EXECUTIVE_SUMMARY = "executive_summary"
    KEY_FINDINGS = "key_findings"
    DETAILED_INSIGHTS = "detailed_insights"
    RECOMMENDATIONS = "recommendations"
    FURTHER_INVESTIGATION = "further_investigation"
    METHODOLOGY = "methodology"
    APPENDIX = "appendix"

class VisualizationType(str, Enum):
    """Types of visualizations that can be suggested."""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEAT_MAP = "heat_map"
    TIME_SERIES = "time_series"
    GEOGRAPHIC_MAP = "geographic_map"
    TABLE = "table"
    TREEMAP = "treemap"
    SANKEY_DIAGRAM = "sankey_diagram"

class Recommendation(BaseModel):
    """Model for an actionable recommendation."""
    recommendation_id: str
    title: str
    description: str
    source_insights: List[str]  # IDs of insights that led to this recommendation
    priority: str  # "high", "medium", or "low"
    implementation_difficulty: str  # "easy", "medium", "hard"
    expected_impact: str
    timeframe: Optional[str] = None

class FurtherInvestigation(BaseModel):
    """Model for an area that needs further investigation."""
    investigation_id: str
    title: str
    description: str
    source_insights: List[str]  # IDs of insights that led to this suggestion
    potential_value: str  # "high", "medium", or "low"
    related_questions: List[str] = Field(default_factory=list)

class Visualization(BaseModel):
    """Model for a visualization suggestion."""
    visualization_id: str
    title: str
    description: str
    visualization_type: VisualizationType
    data_source: str  # Usually an insight ID or query ID
    source_insights: List[str]  # IDs of insights to visualize
    parameters: Dict[str, Any] = Field(default_factory=dict)  # Specific parameters for this visualization

class Report(BaseModel):
    """Model for a complete report."""
    report_id: str
    title: str
    date_generated: str
    original_query: str
    executive_summary: str
    key_findings: List[Dict[str, Any]]  # Simplified top insights
    detailed_insights: Dict[str, List[Dict[str, Any]]]  # Grouped by category
    recommendations: List[Recommendation]
    further_investigations: List[FurtherInvestigation]
    visualizations: List[Visualization]
    methodology: str
    appendix: Optional[Dict[str, Any]] = None

class ReportGeneratingAgent:
    """
    Agent for generating comprehensive reports from consolidated insights.
    
    This agent takes consolidated insights and transforms them into a
    user-friendly report with visualizations and recommendations.
    """
    
    def __init__(self, 
                 openai_model: str = "gpt-4o-mini", 
                 report_format: ReportFormat = ReportFormat.MARKDOWN):
        """
        Initialize the Report Generating Agent.
        
        Args:
            openai_model: OpenAI model to use for report generation
            report_format: Format for the generated report
        """
        self.openai_model = openai_model
        self.report_format = report_format
        
    def _generate_executive_summary(self, 
                                  consolidated_insights: ConsolidatedInsights,
                                  user_query: str) -> str:
        """
        Generate an executive summary based on consolidated insights.
        
        Args:
            consolidated_insights: The consolidated insights
            user_query: The original user query
            
        Returns:
            Executive summary text
        """
        # Prepare insights data for LLM
        top_insights = consolidated_insights.top_insights[:5]  # Limit to top 5 for summary
        top_insights_data = [
            insight.model_dump() for insight in top_insights
        ]
        
        system_prompt = """
        You are an expert data analyst presenting insights to executive stakeholders.
        
        Your task is to create a concise executive summary of the key findings from a data analysis project.
        The summary should:
        
        1. Start with a brief introduction to the analysis scope and objectives
        2. Highlight the most important findings in a clear, non-technical manner
        3. Focus on business implications rather than technical details
        4. Be written in a professional, authoritative tone
        5. Be approximately 2-3 paragraphs in length
        
        Your summary should be immediately valuable to busy executives who need to quickly understand the most important insights.
        """
        
        user_prompt = f"""
        ORIGINAL QUERY: "{user_query}"
        
        TOP INSIGHTS:
        {json.dumps(top_insights_data, indent=2)}
        
        TOTAL INSIGHTS GENERATED: {consolidated_insights.total_insights}
        TOTAL INSIGHT GROUPS: {consolidated_insights.total_groups}
        
        Based on the above information, write a concise executive summary that captures the most important findings
        and their business implications. The summary should be accessible to non-technical stakeholders.
        """
        
        # Call LLM to generate executive summary
        try:
            summary = call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.openai_model,
                temperature=0.7
            )
            
            if summary and isinstance(summary, str):
                return summary
                
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            
        # Fallback if LLM fails
        return f"Analysis of {user_query} identified {consolidated_insights.total_insights} insights across {consolidated_insights.total_groups} thematic groups."
    
    def _format_key_findings(self, 
                           consolidated_insights: ConsolidatedInsights) -> List[Dict[str, Any]]:
        """
        Format top insights into key findings for the report.
        
        Args:
            consolidated_insights: The consolidated insights
            
        Returns:
            Formatted key findings
        """
        key_findings = []
        
        # Use top insights (could be original or synthesized)
        for i, insight in enumerate(consolidated_insights.top_insights[:10]):  # Limit to top 10
            if isinstance(insight, Insight) or isinstance(insight, SynthesizedInsight):
                finding = {
                    "id": f"KF{i+1:02d}",
                    "headline": insight.headline,
                    "description": insight.description,
                    "tier": insight.insight_tier if isinstance(insight.insight_tier, str) else insight.insight_tier.value,
                    "supporting_data": insight.supporting_data
                }
                
                # Add source insight IDs for synthesized insights
                if isinstance(insight, SynthesizedInsight) and hasattr(insight, 'source_insight_ids'):
                    finding["source_insights"] = insight.source_insight_ids
                    
                key_findings.append(finding)
                
        return key_findings
    
    def _group_insights_by_category(self, 
                                  consolidated_insights: ConsolidatedInsights) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group insights by category for detailed section.
        
        Args:
            consolidated_insights: The consolidated insights
            
        Returns:
            Dictionary of insights grouped by category
        """
        grouped_insights = {}
        
        # First, collect all individual insights
        all_insights = []
        
        # Add insights from groups
        for group in consolidated_insights.groups:
            for insight in group.insights:
                all_insights.append((insight, group.group_name))
                
        # Group by category
        for insight, group_name in all_insights:
            # Get category
            category = insight.insight_tier
            if not isinstance(category, str):
                category = category.value
                
            # Initialize category if not exists
            if category not in grouped_insights:
                grouped_insights[category] = []
                
            # Format insight
            formatted_insight = {
                "id": insight.insight_id,
                "headline": insight.headline,
                "description": insight.description,
                "group": group_name,
                "supporting_data": insight.supporting_data
            }
            
            # Add additional fields if they exist
            if hasattr(insight, 'comparison_point') and insight.comparison_point:
                formatted_insight["comparison_point"] = insight.comparison_point
                
            if hasattr(insight, 'significance_metric') and insight.significance_metric:
                formatted_insight["significance_metric"] = insight.significance_metric
                
            if hasattr(insight, 'visualization_suggestion') and insight.visualization_suggestion:
                formatted_insight["visualization_suggestion"] = insight.visualization_suggestion
                
            # Add to category
            grouped_insights[category].append(formatted_insight)
            
        return grouped_insights
    
    def _generate_recommendations(self, 
                                consolidated_insights: ConsolidatedInsights,
                                user_query: str) -> List[Recommendation]:
        """
        Generate actionable recommendations based on insights.
        
        Args:
            consolidated_insights: The consolidated insights
            user_query: The original user query
            
        Returns:
            List of recommendations
        """
        # Prepare insights data for LLM
        top_insights = consolidated_insights.top_insights[:8]  # Limit to top 8 for recommendations
        insights_data = [
            insight.model_dump() for insight in top_insights
        ]
        
        system_prompt = """
        You are an expert data consultant who specializes in converting analytical insights into actionable recommendations.
        
        Your task is to create practical, specific recommendations based on a set of data insights. Each recommendation should:
        
        1. Be directly tied to one or more specific insights
        2. Be actionable and specific, not vague or general
        3. Include a clear title, detailed description, and implementation considerations
        4. Assess priority, implementation difficulty, expected impact, and timeframe
        5. Be relevant to the original business question
        
        Focus on creating recommendations that would provide the most value to the organization if implemented.
        """
        
        user_prompt = f"""
        ORIGINAL QUERY: "{user_query}"
        
        INSIGHTS:
        {json.dumps(insights_data, indent=2)}
        
        Based on these insights, generate 3-5 actionable recommendations.
        
        Format each recommendation as a JSON object with the following structure:
        {{
            "recommendation_id": "R001",  // Unique ID
            "title": "Concise title of the recommendation",
            "description": "Detailed description of what should be done and why",
            "source_insights": ["ID1", "ID2"],  // IDs of insights that support this recommendation
            "priority": "high/medium/low",
            "implementation_difficulty": "easy/medium/hard",
            "expected_impact": "Description of the expected impact",
            "timeframe": "Suggested timeframe for implementation (short-term, medium-term, long-term)"
        }}
        
        Return an array of these recommendation objects.
        """
        
        # Call LLM to generate recommendations
        try:
            recommendations_data = call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=List[Recommendation],
                model=self.openai_model,
                temperature=0.7
            )
            
            if recommendations_data:
                return recommendations_data
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
        # Fallback - empty list
        return []
    
    def _generate_further_investigations(self, 
                                       consolidated_insights: ConsolidatedInsights,
                                       user_query: str) -> List[FurtherInvestigation]:
        """
        Generate suggestions for further investigation based on insights.
        
        Args:
            consolidated_insights: The consolidated insights
            user_query: The original user query
            
        Returns:
            List of further investigation areas
        """
        # Prepare insights data for LLM
        # Use a mix of top insights and some from different groups
        selected_insights = []
        
        # Add top insights
        selected_insights.extend(consolidated_insights.top_insights[:5])
        
        # Add some synthesized insights if available
        if consolidated_insights.synthesized_insights:
            for synth in consolidated_insights.synthesized_insights[:3]:
                if synth not in selected_insights:
                    selected_insights.append(synth)
        
        insights_data = [
            insight.model_dump() for insight in selected_insights
        ]
        
        system_prompt = """
        You are an expert data scientist who specializes in identifying gaps in analyses and opportunities for deeper investigation.
        
        Your task is to identify areas for further investigation based on a set of data insights. Each suggestion should:
        
        1. Highlight a specific question or area that could benefit from deeper analysis
        2. Be connected to one or more existing insights but go beyond them
        3. Include a clear title, detailed description, and potential value
        4. Suggest specific follow-up questions that could be investigated
        5. Be relevant to the original business question
        
        Focus on identifying the most valuable next steps in the analytical journey.
        """
        
        user_prompt = f"""
        ORIGINAL QUERY: "{user_query}"
        
        INSIGHTS:
        {json.dumps(insights_data, indent=2)}
        
        Based on these insights, identify 2-4 areas that warrant further investigation.
        
        Format each suggestion as a JSON object with the following structure:
        {{
            "investigation_id": "F001",  // Unique ID
            "title": "Concise title of the investigation area",
            "description": "Detailed description of what should be investigated and why",
            "source_insights": ["ID1", "ID2"],  // IDs of insights that led to this suggestion
            "potential_value": "high/medium/low",
            "related_questions": ["Specific question 1?", "Specific question 2?"]  // Follow-up questions to investigate
        }}
        
        Return an array of these investigation objects.
        """
        
        # Call LLM to generate further investigations
        try:
            investigations_data = call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=List[FurtherInvestigation],
                model=self.openai_model,
                temperature=0.7
            )
            
            if investigations_data:
                return investigations_data
                
        except Exception as e:
            logger.error(f"Error generating further investigations: {e}")
            
        # Fallback - empty list
        return []
    
    def _generate_visualizations(self, 
                               consolidated_insights: ConsolidatedInsights) -> List[Visualization]:
        """
        Generate visualization suggestions based on insights.
        
        Args:
            consolidated_insights: The consolidated insights
            
        Returns:
            List of visualization suggestions
        """
        visualizations = []
        
        # Create visualizations for top insights first
        for i, insight in enumerate(consolidated_insights.top_insights[:5]):
            if hasattr(insight, 'visualization_suggestion') and insight.visualization_suggestion:
                # Parse the visualization suggestion from the insight
                viz_type = self._parse_visualization_type(insight.visualization_suggestion)
                
                # Create the visualization
                viz = Visualization(
                    visualization_id=f"V{i+1:02d}",
                    title=f"Visualization for {insight.headline}",
                    description=insight.visualization_suggestion,
                    visualization_type=viz_type,
                    data_source=insight.insight_id,
                    source_insights=[insight.insight_id],
                    parameters={}
                )
                
                visualizations.append(viz)
                
        # Add visualizations for synthesized insights
        for i, synth in enumerate(consolidated_insights.synthesized_insights[:3]):
            if hasattr(synth, 'visualization_suggestion') and synth.visualization_suggestion:
                # Parse the visualization suggestion
                viz_type = self._parse_visualization_type(synth.visualization_suggestion)
                
                # Create the visualization
                viz = Visualization(
                    visualization_id=f"VS{i+1:02d}",
                    title=f"Visualization for {synth.headline}",
                    description=synth.visualization_suggestion,
                    visualization_type=viz_type,
                    data_source=synth.synthesis_id,
                    source_insights=synth.source_insight_ids,
                    parameters={}
                )
                
                visualizations.append(viz)
                
        return visualizations
    
    def _parse_visualization_type(self, 
                                viz_suggestion: str) -> VisualizationType:
        """
        Parse visualization type from suggestion text.
        
        Args:
            viz_suggestion: Visualization suggestion text
            
        Returns:
            Visualization type
        """
        suggestion_lower = viz_suggestion.lower()
        
        # Map common terms to visualization types
        if "bar" in suggestion_lower:
            return VisualizationType.BAR_CHART
        elif "line" in suggestion_lower:
            return VisualizationType.LINE_CHART
        elif "pie" in suggestion_lower:
            return VisualizationType.PIE_CHART
        elif "scatter" in suggestion_lower:
            return VisualizationType.SCATTER_PLOT
        elif "heat" in suggestion_lower or "map" in suggestion_lower:
            return VisualizationType.HEAT_MAP
        elif "time" in suggestion_lower or "trend" in suggestion_lower:
            return VisualizationType.TIME_SERIES
        elif "geo" in suggestion_lower or "map" in suggestion_lower:
            return VisualizationType.GEOGRAPHIC_MAP
        elif "table" in suggestion_lower:
            return VisualizationType.TABLE
        elif "tree" in suggestion_lower:
            return VisualizationType.TREEMAP
        elif "sankey" in suggestion_lower or "flow" in suggestion_lower:
            return VisualizationType.SANKEY_DIAGRAM
        
        # Default to bar chart if can't determine
        return VisualizationType.BAR_CHART
    
    def _generate_methodology(self, 
                            original_query: str, 
                            consolidated_insights: ConsolidatedInsights) -> str:
        """
        Generate methodology section for the report.
        
        Args:
            original_query: The original user query
            consolidated_insights: The consolidated insights
            
        Returns:
            Methodology text
        """
        methodology = f"""
## Methodology

This analysis was conducted using the Data Insights Generator automated analysis system. The process involved:

1. **Query Understanding**: The system analyzed the question "{original_query}" to determine appropriate analytical approaches.

2. **Data Exploration**: The system explored the database structure and sampled data to understand available information.

3. **Question Generation**: Based on the initial query, the system generated {consolidated_insights.total_insights} potential analytical questions.

4. **Data Analysis**: SQL queries were automatically generated and executed to answer these questions.

5. **Insight Generation**: The system analyzed query results to identify {consolidated_insights.total_insights} distinct insights.

6. **Insight Consolidation**: Similar insights were grouped into {consolidated_insights.total_groups} thematic clusters and synthesized into higher-level findings.

7. **Report Generation**: This report was automatically generated based on the most significant findings, with recommendations and visualization suggestions derived from the insights.

All analytical steps were performed using a combination of automated data analysis and large language models trained on data analysis best practices.
"""
        return methodology
    
    def _format_report_markdown(self, report: Report) -> str:
        """
        Format the report as Markdown.
        
        Args:
            report: The report to format
            
        Returns:
            Markdown-formatted report
        """
        markdown_content = f"""
# {report.title}

*Report generated on {report.date_generated}*

## Executive Summary

{report.executive_summary}

## Key Findings

"""
        # Add key findings
        for finding in report.key_findings:
            markdown_content += f"### {finding['headline']}\n\n"
            markdown_content += f"{finding['description']}\n\n"
            
            # Add supporting data if available
            if finding.get('supporting_data'):
                markdown_content += "**Supporting Data:**\n\n"
                for key, value in finding['supporting_data'].items():
                    markdown_content += f"- {key}: {value}\n"
                markdown_content += "\n"
                
        # Add detailed insights section
        markdown_content += "## Detailed Insights\n\n"
        
        for category, insights in report.detailed_insights.items():
            markdown_content += f"### {category.title()} Insights\n\n"
            
            for insight in insights:
                markdown_content += f"#### {insight['headline']}\n\n"
                markdown_content += f"{insight['description']}\n\n"
                
                # Add group info
                markdown_content += f"*From group: {insight['group']}*\n\n"
                
                # Add supporting data if available
                if insight.get('supporting_data'):
                    markdown_content += "**Supporting Data:**\n\n"
                    for key, value in insight['supporting_data'].items():
                        markdown_content += f"- {key}: {value}\n"
                    markdown_content += "\n"
                    
                # Add visualization suggestion if available
                if insight.get('visualization_suggestion'):
                    markdown_content += f"**Visualization Suggestion:** {insight['visualization_suggestion']}\n\n"
                    
        # Add recommendations section
        markdown_content += "## Recommendations\n\n"
        
        for rec in report.recommendations:
            markdown_content += f"### {rec.title}\n\n"
            markdown_content += f"{rec.description}\n\n"
            
            markdown_content += f"**Priority:** {rec.priority.upper()} | "
            markdown_content += f"**Difficulty:** {rec.implementation_difficulty.title()} | "
            markdown_content += f"**Timeframe:** {rec.timeframe}\n\n"
            
            markdown_content += f"**Expected Impact:** {rec.expected_impact}\n\n"
            
            if rec.source_insights:
                markdown_content += f"*Based on insights: {', '.join(rec.source_insights)}*\n\n"
                
        # Add further investigation section
        markdown_content += "## Areas for Further Investigation\n\n"
        
        for investigation in report.further_investigations:
            markdown_content += f"### {investigation.title}\n\n"
            markdown_content += f"{investigation.description}\n\n"
            
            markdown_content += f"**Potential Value:** {investigation.potential_value.upper()}\n\n"
            
            if investigation.related_questions:
                markdown_content += "**Questions to Explore:**\n\n"
                for question in investigation.related_questions:
                    markdown_content += f"- {question}\n"
                markdown_content += "\n"
                
            if investigation.source_insights:
                markdown_content += f"*Based on insights: {', '.join(investigation.source_insights)}*\n\n"
                
        # Add visualizations section
        markdown_content += "## Suggested Visualizations\n\n"
        
        for viz in report.visualizations:
            markdown_content += f"### {viz.title}\n\n"
            markdown_content += f"**Type:** {viz.visualization_type.value.replace('_', ' ').title()}\n\n"
            markdown_content += f"{viz.description}\n\n"
            
            if viz.source_insights:
                markdown_content += f"*Based on insights: {', '.join(viz.source_insights)}*\n\n"
                
        # Add methodology section
        markdown_content += report.methodology
        
        # Add appendix if available
        if report.appendix:
            markdown_content += "\n## Appendix\n\n"
            
            for title, content in report.appendix.items():
                markdown_content += f"### {title}\n\n"
                markdown_content += f"{content}\n\n"
                
        return markdown_content
    
    def _format_report_html(self, markdown_content: str) -> str:
        """
        Convert Markdown report to HTML.
        
        Args:
            markdown_content: Markdown content
            
        Returns:
            HTML-formatted report
        """
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
        
        # Add basic styling
        styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Insights Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            margin-top: 1.5em;
        }}
        h1 {{
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f9f9f9;
            padding: 3px 5px;
            border-radius: 3px;
            font-family: monospace;
        }}
        blockquote {{
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-left: 0;
            color: #555;
        }}
        .priority-high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .priority-medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .priority-low {{
            color: #27ae60;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
        return styled_html
    
    def _format_report(self, report: Report) -> str:
        """
        Format the report in the specified format.
        
        Args:
            report: The report to format
            
        Returns:
            Formatted report content
        """
        if self.report_format == ReportFormat.MARKDOWN:
            return self._format_report_markdown(report)
        elif self.report_format == ReportFormat.HTML:
            markdown_content = self._format_report_markdown(report)
            return self._format_report_html(markdown_content)
        elif self.report_format == ReportFormat.JSON:
            return json.dumps(report.model_dump(), indent=2)
        else:
            # Default to markdown
            return self._format_report_markdown(report)
    
    def generate_report(self, 
                       consolidated_insights: ConsolidatedInsights,
                       user_query: str,
                       report_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report from consolidated insights.
        
        Args:
            consolidated_insights: The consolidated insights
            user_query: The original user query
            report_title: Optional title for the report (if None, will be generated)
            
        Returns:
            Dictionary with report content and metadata
        """
        # Generate report title if not provided
        title = report_title
        if not title:
            title = f"Data Insights Report: {user_query}"
            
        # Generate date
        date_generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate report ID
        report_id = f"R{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(consolidated_insights, user_query)
        
        # Format key findings
        key_findings = self._format_key_findings(consolidated_insights)
        
        # Group insights by category
        detailed_insights = self._group_insights_by_category(consolidated_insights)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(consolidated_insights, user_query)
        
        # Generate further investigation suggestions
        further_investigations = self._generate_further_investigations(consolidated_insights, user_query)
        
        # Generate visualization suggestions
        visualizations = self._generate_visualizations(consolidated_insights)
        
        # Generate methodology section
        methodology = self._generate_methodology(user_query, consolidated_insights)
        
        # Create the report
        report = Report(
            report_id=report_id,
            title=title,
            date_generated=date_generated,
            original_query=user_query,
            executive_summary=executive_summary,
            key_findings=key_findings,
            detailed_insights=detailed_insights,
            recommendations=recommendations,
            further_investigations=further_investigations,
            visualizations=visualizations,
            methodology=methodology
        )
        
        # Format the report
        formatted_report = self._format_report(report)
        
        # Return the result
        return {
            "report": report.model_dump(),
            "formatted_report": formatted_report,
            "format": self.report_format.value
        }

def main(consolidated_insights_path: str, user_query: str, output_format: str = "markdown", output_dir: str = "./reports"):
    """
    Entry point function to run the Report Generating Agent.
    
    Args:
        consolidated_insights_path: Path to consolidated insights JSON file
        user_query: Original user query
        output_format: Format for the report ("markdown", "html", or "json")
        output_dir: Directory to save the report
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load consolidated insights
    with open(consolidated_insights_path, 'r') as f:
        consolidated_insights_data = json.load(f)
    
    # Convert to ConsolidatedInsights model
    consolidated_insights = ConsolidatedInsights(**consolidated_insights_data)
    
    # Determine report format
    try:
        report_format = ReportFormat(output_format.lower())
    except ValueError:
        logger.warning(f"Invalid report format: {output_format}. Using 'markdown' instead.")
        report_format = ReportFormat.MARKDOWN
    
    # Initialize the Report Agent
    report_agent = ReportGeneratingAgent(
        openai_model="gpt-4o-mini",
        report_format=report_format
    )
    
    # Generate the report
    report = report_agent.generate_report(
        consolidated_insights=consolidated_insights,
        user_query=user_query,
    )

    # Save the report
    report_path = os.path.join(output_dir, f"{user_query}.{report_format.value}")
    with open(report_path, 'w') as f:
        f.write(report['formatted_report'])

    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python report_agent.py <consolidated_insights_path> <user_query> [output_format] [output_dir]")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])    