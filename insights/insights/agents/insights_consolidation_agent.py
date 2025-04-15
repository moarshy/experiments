import json
import logging
from typing import Dict, Any, List, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

from insights.llm import call_openai_api
from insights.utils import setup_logging
from insights.agents.insight_agent import InsightTier, Insight

# Set up NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = setup_logging()

class InsightGroup(BaseModel):
    """Model for a group of related insights."""
    group_id: str
    group_name: str
    insights: List[Insight]
    synthesis: Optional[str] = None  # Synthesized narrative for the group
    priority_score: float = Field(ge=0.0, le=1.0)  # Overall priority of this group
    related_groups: List[str] = Field(default_factory=list)  # IDs of related groups
    
class SynthesizedInsight(BaseModel):
    """Model for a synthesized insight created from multiple related insights."""
    synthesis_id: str
    source_insight_ids: List[str]  # Original insights that were synthesized
    insight_tier: InsightTier  # Usually higher tier than source insights
    headline: str  # Concise summary of the synthesized insight
    description: str  # Comprehensive explanation with context
    supporting_data: Dict[str, Any] = Field(default_factory=dict)  # Key evidence from source insights
    visualization_suggestion: Optional[str] = None  # Suggested visualization type
    priority_score: float = Field(ge=0.0, le=1.0)  # Priority score for this synthesis

class ConsolidatedInsights(BaseModel):
    """Model for the complete set of consolidated insights."""
    original_query: str  # The original user query
    total_insights: int  # Total number of original insights
    total_groups: int  # Total number of insight groups
    total_synthesized: int  # Total number of synthesized insights
    groups: List[InsightGroup]  # Groups of related insights
    synthesized_insights: List[SynthesizedInsight]  # Higher-level synthesized insights
    top_insights: List[Union[Insight, SynthesizedInsight]]  # Top overall insights (prioritized)
    tier_distribution: Dict[str, int] = Field(default_factory=dict)  # Distribution by tier

class NLTKTextProcessor:
    """
    Class for processing text using NLTK for similarity comparison.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for similarity comparison.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                          if token not in self.stop_words and token.isalpha()]
        
        # Join back into a string
        return ' '.join(filtered_tokens)
    
    def compute_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts.
        
        Args:
            texts: List of texts to compare
            
        Returns:
            Similarity matrix as a numpy array
        """
        if not texts:
            return np.array([])
            
        # Create or update the vectorizer
        self.vectorizer = TfidfVectorizer()
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Compute cosine similarity
        return cosine_similarity(tfidf_matrix)

class InsightConsolidationAgent:
    """
    Agent for consolidating, deduplicating, and synthesizing insights.
    
    This agent takes multiple insights, identifies similarities,
    groups related insights, and synthesizes them into higher-level findings.
    """
    
    def __init__(self, 
                 openai_model: str = "gpt-4o",
                 similarity_threshold: float = 0.75,
                 max_top_insights: int = 10):
        """
        Initialize the Insight Consolidation Agent.
        
        Args:
            openai_model: OpenAI model to use for synthesis
            similarity_threshold: Threshold for considering insights similar
            max_top_insights: Maximum number of top insights to include
        """
        self.openai_model = openai_model
        self.similarity_threshold = similarity_threshold
        self.max_top_insights = max_top_insights
        self.text_processor = NLTKTextProcessor()
        
    def _extract_all_insights(self, insights_data: Dict[str, Any]) -> List[Insight]:
        """
        Extract all individual insights from the insights data.
        
        Args:
            insights_data: Data from the Insight Agent
            
        Returns:
            List of all individual insights
        """
        all_insights = []
        
        insight_results = insights_data.get("insights", [])
        for result in insight_results:
            insights = result.get("insights", [])
            for insight in insights:
                # Convert dict to Insight model
                try:
                    insight_model = Insight(**insight)
                    all_insights.append(insight_model)
                except Exception as e:
                    logger.error(f"Error parsing insight: {e}")
                    continue
        
        return all_insights
    
    def _calculate_insight_similarity(self, insights: List[Insight]) -> np.ndarray:
        """
        Calculate pairwise similarity between insights.
        
        Args:
            insights: List of insights
            
        Returns:
            Similarity matrix
        """
        if not insights:
            return np.array([])
            
        # Combine headline and description for similarity comparison
        texts = []
        for insight in insights:
            text = f"{insight.headline} {insight.description}"
            processed_text = self.text_processor.preprocess_text(text)
            texts.append(processed_text)
            
        return self.text_processor.compute_similarity(texts)
    
    def _deduplicate_insights(self, insights: List[Insight]) -> List[Insight]:
        """
        Remove duplicate or very similar insights.
        
        Args:
            insights: List of insights
            
        Returns:
            Deduplicated list of insights
        """
        if not insights:
            return []
            
        # Calculate similarity matrix
        similarity_matrix = self._calculate_insight_similarity(insights)
        
        # Sort insights by tier and confidence for deduplication priority
        tier_priority = {
            InsightTier.CONTRIBUTION: 5,
            InsightTier.PATTERN: 4,
            InsightTier.SIGNIFICANCE: 3,
            InsightTier.COMPARISON: 2,
            InsightTier.OBSERVATION: 1
        }
        
        indexed_insights = list(enumerate(insights))
        sorted_indexed_insights = sorted(
            indexed_insights, 
            key=lambda x: (tier_priority.get(x[1].insight_tier, 0), x[1].confidence_score),
            reverse=True
        )
        
        # Deduplicate
        unique_indices = set()
        duplicate_groups = []
        
        for i, (orig_idx, insight) in enumerate(sorted_indexed_insights):
            # Skip if already processed as a duplicate
            if orig_idx in unique_indices:
                continue
                
            # Start a new group with this insight
            group = [orig_idx]
            unique_indices.add(orig_idx)
            
            # Find similar insights
            for j, (other_idx, _) in enumerate(sorted_indexed_insights):
                if other_idx == orig_idx or other_idx in unique_indices:
                    continue
                    
                if similarity_matrix[orig_idx, other_idx] >= self.similarity_threshold:
                    group.append(other_idx)
                    unique_indices.add(other_idx)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        # Keep only the unique insights
        unique_insights = []
        for i, insight in enumerate(insights):
            if i in unique_indices:
                unique_insights.append(insight)
                
        logger.info(f"Deduplicated {len(insights)} insights to {len(unique_insights)} unique insights")
        return unique_insights
    
    def _cluster_insights(self, insights: List[Insight]) -> List[List[int]]:
        """
        Cluster insights into related groups using similarity matrix and graph community detection.
        
        Args:
            insights: List of insights
            
        Returns:
            List of cluster indices
        """
        if not insights:
            return []
            
        # Calculate similarity matrix
        similarity_matrix = self._calculate_insight_similarity(insights)
        
        # Create a graph where nodes are insights and edges are weighted by similarity
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(insights)):
            G.add_node(i)
            
        # Add edges for similar insights (above threshold)
        for i in range(len(insights)):
            for j in range(i+1, len(insights)):
                similarity = similarity_matrix[i, j]
                if similarity >= self.similarity_threshold * 0.8:  # Lower threshold for clustering
                    G.add_edge(i, j, weight=similarity)
        
        # Use community detection to find clusters
        clusters = []
        
        # Handle disconnected graph
        if nx.number_connected_components(G) > 1:
            # Get connected components
            for component in nx.connected_components(G):
                clusters.append(list(component))
        else:
            # Use Louvain community detection
            try:
                from community import best_partition
                partition = best_partition(G)
                
                # Group by community
                community_to_nodes = {}
                for node, community_id in partition.items():
                    if community_id not in community_to_nodes:
                        community_to_nodes[community_id] = []
                    community_to_nodes[community_id].append(node)
                
                clusters = list(community_to_nodes.values())
            except:
                # Fallback to a simpler approach
                clusters = [list(range(len(insights)))]  # All in one cluster
        
        # Handle case with no clustering
        if not clusters:
            clusters = [[i] for i in range(len(insights))]  # Each insight in its own cluster
            
        return clusters
    
    def _generate_group_name(self, insights: List[Insight]) -> str:
        """
        Generate a descriptive name for a group of insights.
        
        Args:
            insights: List of insights in the group
            
        Returns:
            Descriptive name for the group
        """
        if not insights:
            return "Empty Group"
            
        # Extract keywords from insights
        all_text = " ".join([f"{insight.headline} {insight.description}" for insight in insights])
        processed_text = self.text_processor.preprocess_text(all_text)
        
        # Use TFIDF to find important terms
        vectorizer = TfidfVectorizer(max_features=10)
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_indices = tfidf_scores.argsort()[-3:][::-1]  # Top 3 terms
            top_terms = [feature_names[i] for i in top_indices]
            
            # Create group name
            group_name = f"Group: {', '.join(top_terms).title()}"
            return group_name
        except:
            # Fallback to simpler approach
            # Find the highest tier insight
            tier_priority = {
                InsightTier.CONTRIBUTION: 5,
                InsightTier.PATTERN: 4,
                InsightTier.SIGNIFICANCE: 3,
                InsightTier.COMPARISON: 2,
                InsightTier.OBSERVATION: 1
            }
            
            sorted_insights = sorted(
                insights, 
                key=lambda x: tier_priority.get(x.insight_tier, 0),
                reverse=True
            )
            
            # Use headline of highest tier insight
            top_insight = sorted_insights[0]
            return f"Group: {top_insight.headline[:30]}..."
    
    def _calculate_priority_score(self, 
                                 insight: Union[Insight, SynthesizedInsight],
                                 query_keywords: Set[str]) -> float:
        """
        Calculate priority score for an insight based on tier, confidence, and relevance to query.
        
        Args:
            insight: The insight to score
            query_keywords: Keywords from the original query
            
        Returns:
            Priority score between 0.0 and 1.0
        """
        # Base weights
        tier_weights = {
            InsightTier.CONTRIBUTION: 1.0,
            InsightTier.PATTERN: 0.8,
            InsightTier.SIGNIFICANCE: 0.7,
            InsightTier.COMPARISON: 0.5,
            InsightTier.OBSERVATION: 0.3
        }
        
        # Base score from tier
        tier_score = tier_weights.get(insight.insight_tier, 0.5)
        
        # Confidence score if available (for original insights)
        confidence_score = 0.8  # Default for synthesized insights
        if hasattr(insight, 'confidence_score'):
            confidence_score = insight.confidence_score
            
        # Keyword relevance
        relevance_score = 0.0
        if query_keywords:
            # Create text from insight
            insight_text = f"{insight.headline} {insight.description}".lower()
            insight_words = set(self.text_processor.preprocess_text(insight_text).split())
            
            # Count keyword matches
            matches = insight_words.intersection(query_keywords)
            if matches:
                relevance_score = len(matches) / len(query_keywords)
        
        # Combine scores (with weights)
        final_score = (0.4 * tier_score) + (0.3 * confidence_score) + (0.3 * relevance_score)
        
        return min(max(final_score, 0.0), 1.0)  # Ensure between 0 and 1
    
    def _synthesize_insights(self, 
                           group: List[Insight], 
                           group_id: str) -> Optional[SynthesizedInsight]:
        """
        Synthesize a group of related insights into a higher-level insight.
        
        Args:
            group: List of related insights
            group_id: ID of the group
            
        Returns:
            Synthesized insight or None if synthesis failed
        """
        if len(group) <= 1:
            return None  # No need to synthesize a single insight
            
        # Prepare insights for LLM
        insights_data = [insight.model_dump() for insight in group]
        
        system_prompt = """
        You are an expert data analyst who specializes in synthesizing related insights into higher-level findings.
        
        Your task is to analyze multiple related insights and create a synthesized insight that:
        1. Captures the most important aspects of all source insights
        2. Provides a higher-level perspective that connects the individual insights
        3. Draws a more significant conclusion than any individual insight
        
        The synthesized insight should:
        - Have a tier that is at least as high as the highest tier in the source insights (ideally higher)
        - Include a concise headline that captures the essence of the synthesis
        - Provide a comprehensive description that connects all the source insights
        - Reference supporting data from the source insights
        - Suggest an appropriate visualization if applicable
        
        Your synthesis should be more valuable and insightful than the sum of the individual insights.
        """
        
        user_prompt = f"""
        SOURCE INSIGHTS:
        {json.dumps(insights_data, indent=2)}
        
        Based on these related insights, create a synthesized higher-level insight that connects them and draws a more significant conclusion.
        
        Format your response as a JSON object with the following structure:
        {{
            "synthesis_id": "{group_id}-S001",
            "source_insight_ids": ["list of IDs of the source insights"],
            "insight_tier": "one of: observation, comparison, significance, pattern, contribution",
            "headline": "Concise headline of the synthesized insight",
            "description": "Comprehensive description that connects all source insights",
            "supporting_data": {{ "key evidence": "values" }},
            "visualization_suggestion": "Suggested visualization type" or null if not applicable
        }}
        """
        
        # Call LLM to synthesize insights
        try:
            synthesis_data = call_openai_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=SynthesizedInsight,
                model=self.openai_model,
                temperature=0.5
            )
            
            if synthesis_data:
                return synthesis_data
                
        except Exception as e:
            logger.error(f"Error synthesizing insights: {e}")
            
        return None
    
    def _identify_related_groups(self, groups: List[InsightGroup]) -> List[InsightGroup]:
        """
        Identify relationships between insight groups.
        
        Args:
            groups: List of insight groups
            
        Returns:
            Updated groups with related_groups field populated
        """
        if len(groups) <= 1:
            return groups
            
        # Extract group texts
        texts = []
        for group in groups:
            # Combine all insight texts in the group
            group_text = " ".join([
                f"{insight.headline} {insight.description}" 
                for insight in group.insights
            ])
            
            # Add synthesis if available
            if group.synthesis:
                group_text += f" {group.synthesis}"
                
            # Process text
            processed_text = self.text_processor.preprocess_text(group_text)
            texts.append(processed_text)
            
        # Calculate similarity matrix
        similarity_matrix = self.text_processor.compute_similarity(texts)
        
        # Find related groups
        threshold = self.similarity_threshold * 0.7  # Lower threshold for group relations
        
        for i, group in enumerate(groups):
            related = []
            for j, other_group in enumerate(groups):
                if i != j and similarity_matrix[i, j] >= threshold:
                    related.append(other_group.group_id)
            
            group.related_groups = related
            
        return groups
        
    def consolidate_insights(self, 
                           insights_data: Dict[str, Any],
                           original_query: str) -> ConsolidatedInsights:
        """
        Consolidate, deduplicate, and synthesize insights.
        
        Args:
            insights_data: Data from the Insight Agent
            original_query: The original user query
            
        Returns:
            Consolidated insights
        """
        # Extract all insights
        all_insights = self._extract_all_insights(insights_data)
        
        if not all_insights:
            logger.warning("No insights to consolidate")
            return ConsolidatedInsights(
                original_query=original_query,
                total_insights=0,
                total_groups=0,
                total_synthesized=0,
                groups=[],
                synthesized_insights=[],
                top_insights=[]
            )
            
        # Deduplicate insights
        unique_insights = self._deduplicate_insights(all_insights)
        
        # Preprocess query for relevance scoring
        query_keywords = set(self.text_processor.preprocess_text(original_query).split())
        
        # Cluster insights into groups
        cluster_indices = self._cluster_insights(unique_insights)
        
        # Create insight groups
        groups = []
        synthesized_insights = []
        
        for i, cluster in enumerate(cluster_indices):
            group_id = f"G{str(i+1).zfill(3)}"
            
            # Get insights in this cluster
            cluster_insights = [unique_insights[idx] for idx in cluster]
            
            # Generate group name
            group_name = self._generate_group_name(cluster_insights)
            
            # Synthesize insights if more than one
            synthesis = None
            if len(cluster_insights) > 1:
                synthesized = self._synthesize_insights(cluster_insights, group_id)
                if synthesized:
                    synthesis = synthesized.description
                    
                    # Calculate priority score
                    synthesized.priority_score = self._calculate_priority_score(synthesized, query_keywords)
                    
                    # Add to synthesized insights
                    synthesized_insights.append(synthesized)
            
            # Calculate group priority score
            priority_scores = [self._calculate_priority_score(insight, query_keywords) 
                              for insight in cluster_insights]
            group_priority = max(priority_scores) if priority_scores else 0.0
            
            # Create group
            group = InsightGroup(
                group_id=group_id,
                group_name=group_name,
                insights=cluster_insights,
                synthesis=synthesis,
                priority_score=group_priority,
                related_groups=[]  # Will be populated later
            )
            
            groups.append(group)
            
        # Identify related groups
        groups = self._identify_related_groups(groups)
        
        # Select top insights
        all_candidate_insights = []
        
        # Add synthesized insights
        for synthesized in synthesized_insights:
            all_candidate_insights.append(synthesized)
            
        # Add original insights
        for insight in unique_insights:
            # Calculate priority score
            priority_score = self._calculate_priority_score(insight, query_keywords)
            
            # Create a wrapper with priority score
            insight_dict = insight.model_dump()
            insight_dict["priority_score"] = priority_score
            
            # Re-create to add priority_score
            updated_insight = Insight(**insight_dict)
            
            all_candidate_insights.append(updated_insight)
            
        # Sort by priority score
        sorted_insights = sorted(
            all_candidate_insights,
            key=lambda x: x.priority_score,
            reverse=True
        )
        
        # Select top N
        top_insights = sorted_insights[:self.max_top_insights]
        
        # Calculate tier distribution
        tier_distribution = {}
        for insight in unique_insights:
            tier = insight.insight_tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            
        # Create final result
        result = ConsolidatedInsights(
            original_query=original_query,
            total_insights=len(all_insights),
            total_groups=len(groups),
            total_synthesized=len(synthesized_insights),
            groups=groups,
            synthesized_insights=synthesized_insights,
            top_insights=top_insights,
            tier_distribution=tier_distribution
        )
        
        return result

def main(insights_path: str, original_query: str, output_path: str = "consolidated_insights.json"):
    """
    Entry point function to run the Insight Consolidation Agent.
    
    Args:
        insights_path: Path to insights JSON file from Insight Agent
        original_query: Original user query
        output_path: Path to save consolidated insights JSON file
    """
    # Load insights
    with open(insights_path, 'r') as f:
        insights_data = json.load(f)
    
    # Create agent
    agent = InsightConsolidationAgent()
    
    # Consolidate insights
    consolidated = agent.consolidate_insights(insights_data, original_query)
    
    # Save consolidated insights
    with open(output_path, 'w') as f:
        json.dump(consolidated.model_dump(), f, indent=2)
    
    logger.info(f"Consolidated {consolidated.total_insights} insights into {consolidated.total_groups} groups with {consolidated.total_synthesized} synthesized insights. Saved to {output_path}")
    
    # Print top insights
    print(f"\nTop {len(consolidated.top_insights)} insights:")
    for i, insight in enumerate(consolidated.top_insights):
        if hasattr(insight, 'headline'):  # Both Insight and SynthesizedInsight have headline
            print(f"{i+1}. [{getattr(insight, 'insight_tier', 'unknown')}] {insight.headline}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python insight_consolidation_agent.py <insights_path> <original_query> [output_path]")
        sys.exit(1)
    
    insights_path = sys.argv[1]
    original_query = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "consolidated_insights.json"
    
    main(insights_path, original_query, output_path)