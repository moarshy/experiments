import json
import time
import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
import numpy as np
import openai

from insights.agents.insight_agent import StructuredInsight, InsightTier
from insights.llm import LLM, OpenAIClient # Import specific client if needed
from insights.agents.db_summary_agent import DatabaseSummary
from insights.config import OPENAI_API_KEY # Needed for OpenAI embeddings

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from insights.utils import setup_logging

logger = setup_logging(__name__)


# --- Agent Configuration Defaults ---
DEFAULT_DEDUPLICATION_THRESHOLD = 0.90
DEFAULT_SYNTHESIS_CLUSTER_THRESHOLD = 0.75
DEFAULT_PRIORITY_WEIGHTS = {"relevance": 0.4, "significance": 0.3, "confidence": 0.1, "tier": 0.2}
DEFAULT_TIER_SCORES = { InsightTier.ANOMALY: 1.0, InsightTier.CONTRIBUTION: 0.9, InsightTier.RELATIONSHIP: 0.8, InsightTier.TREND: 0.7, InsightTier.COMPARISON: 0.6, InsightTier.OBSERVATION: 0.3, InsightTier.DATA_QUALITY: 0.2, InsightTier.EXECUTION_INFO: 0.1 }
DEFAULT_FILTER_TOP_N = 10
DEFAULT_OPENAI_EMBEDDING_MODEL = 'text-embedding-3-large' 

# --- Insight Consolidation Agent ---
class InsightConsolidationAgent:
    """
    Consolidates raw insights using OpenAI embeddings, deduplication,
    prioritization, synthesis, and filtering.
    """
    def __init__(self,
                 llm_provider: str = "openai",
                 embedding_model_name: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
                 deduplication_threshold: float = DEFAULT_DEDUPLICATION_THRESHOLD,
                 synthesis_cluster_threshold: float = DEFAULT_SYNTHESIS_CLUSTER_THRESHOLD,
                 priority_weights: Dict[str, float] = None,
                 tier_scores: Dict[InsightTier, float] = None,
                 filter_top_n: Optional[int] = None,
                 enable_synthesis: bool = True):
        """
        Initializes the Insight Consolidation Agent using OpenAI embeddings.

        Args:
            llm_provider: LLM provider for synthesis ('openai' or 'gemini').
            llm_model_config: Config for the synthesis LLM client.
            embedding_model_name: Specific OpenAI embedding model name.
            deduplication_threshold: Similarity threshold for deduplication.
            synthesis_cluster_threshold: Similarity threshold for synthesis clustering.
            priority_weights: Weights for ranking factors.
            tier_scores: Scores for insight tiers.
            filter_top_n: Keep top N insights (None to disable).
            enable_synthesis: Flag to enable/disable synthesis step.
        """
        # --- LLM Client for Synthesis ---
        self.llm_client = LLM.create_client(provider=llm_provider)
        if self.llm_client is None:
             raise ValueError(f"Could not create LLM client for provider: {llm_provider}")

        # --- Embedding Client (OpenAI Only) ---
        self.embedding_model_name = embedding_model_name
        self.openai_embedding_client = None

        if isinstance(self.llm_client, OpenAIClient) and hasattr(self.llm_client, 'client'):
             # Reuse the client initialized for generation if it's OpenAI
             self.openai_embedding_client = self.llm_client.client # Access the underlying openai.OpenAI client
             logger.info(f"Using existing OpenAI client for embeddings (Model: {self.embedding_model_name}).")
        else:
             self.openai_embedding_client = openai.OpenAI(api_key=OPENAI_API_KEY)
             logger.info(f"Created separate OpenAI client for embeddings (Model: {self.embedding_model_name}).")
        if self.openai_embedding_client is None:
             raise ValueError("Failed to initialize OpenAI client for embeddings.")

        # --- Other Configs ---
        self.dedup_threshold = deduplication_threshold
        self.synth_cluster_threshold = synthesis_cluster_threshold
        self.priority_weights = priority_weights or DEFAULT_PRIORITY_WEIGHTS
        self.tier_scores = tier_scores or DEFAULT_TIER_SCORES
        self.filter_top_n = filter_top_n
        self.enable_synthesis = enable_synthesis

        logger.info(f"InsightConsolidationAgent initialized. Embedding Provider: OpenAI (Model: {self.embedding_model_name}), Deduplication: {self.dedup_threshold}, Synthesis: {self.enable_synthesis}")

    # --- Main Orchestration Method ---
    def consolidate(self,
                    insights_list: List[StructuredInsight],
                    original_user_query: Optional[str] = None,
                    db_summary: Optional[DatabaseSummary] = None) -> List[StructuredInsight]:
        """Orchestrates the insight consolidation process."""
        logger.info(f"Starting consolidation for {len(insights_list)} raw insights...")
        # 1. Deduplication
        unique_insights = self._deduplicate_insights(insights_list)
        logger.info(f"Reduced to {len(unique_insights)} unique insights after deduplication.")
        processed_insights = unique_insights
        # 2. Synthesis (Optional)
        if self.enable_synthesis and len(unique_insights) > 1:
            logger.info("Attempting insight synthesis...")
            clustered_insights, unclustered_insights = self._cluster_insights(unique_insights)
            logger.info(f"Found {len(clustered_insights)} clusters for synthesis. {len(unclustered_insights)} insights remain unclustered.")
            synthesized_insights = self._synthesize_clusters(clustered_insights, original_user_query, db_summary)
            logger.info(f"Generated {len(synthesized_insights)} synthesized insights.")
            processed_insights = synthesized_insights + unclustered_insights # Combine synthesized with non-clustered originals
            logger.info(f"Total insights after synthesis step: {len(processed_insights)}")
        # 3. Prioritization
        prioritized_insights = self._prioritize_insights(processed_insights)
        logger.info(f"Prioritized {len(prioritized_insights)} insights.")
        # 4. Filtering
        final_insights = self._filter_insights(prioritized_insights)
        logger.info(f"Filtered down to {len(final_insights)} final insights.")
        return final_insights

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings using the configured OpenAI model."""
        if not texts:
            return np.array([])
        if not self.openai_embedding_client:
             raise ValueError("OpenAI client for embeddings not initialized.")

        logger.debug(f"Generating OpenAI embeddings for {len(texts)} texts using {self.embedding_model_name}...")
        try:
            # Replace empty strings or None with a placeholder if necessary,
            # as OpenAI API might error on empty input strings.
            processed_texts = [text if text and text.strip() else " " for text in texts]

            response = self.openai_embedding_client.embeddings.create(
                input=processed_texts,
                model=self.embedding_model_name
            )
            embeddings_list = [item.embedding for item in response.data]

            # Validation
            if not embeddings_list:
                 logger.warning("OpenAI embedding API returned no embeddings.")
                 return np.array([])
            if len(embeddings_list) != len(texts):
                 logger.error(f"Mismatch in number of embeddings returned ({len(embeddings_list)}) vs texts sent ({len(texts)}).")
                 # Handle mismatch - maybe return empty or raise error depending on desired robustness
                 raise ValueError("OpenAI embedding count mismatch.")
            dim = len(embeddings_list[0])
            if not all(len(e) == dim for e in embeddings_list):
                 logger.error("OpenAI embeddings have inconsistent dimensions.")
                 raise ValueError("Inconsistent embedding dimensions from OpenAI.")

            return np.array(embeddings_list)
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}", exc_info=True)
            raise # Propagate error

    def _get_insight_text(self, insight: StructuredInsight) -> str:
        """Combines key text fields for embedding."""
        return f"{insight.headline}. {insight.description}"

    def _deduplicate_insights(self, insights: List[StructuredInsight]) -> List[StructuredInsight]:
        """Identifies and merges semantically similar insights using OpenAI embeddings."""
        if len(insights) <= 1: return insights
        texts_to_embed = [self._get_insight_text(i) for i in insights]
        try:
             embeddings = self._get_embeddings(texts_to_embed) # Use helper method
        except Exception as e:
             logger.error(f"Failed to get embeddings for deduplication: {e}. Skipping deduplication.")
             return insights # Return original list if embedding fails

        if embeddings.shape[0] != len(insights):
             logger.error("Embedding count mismatch. Skipping deduplication.")
             return insights

        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)
        merged_indices = set()
        unique_insights_list = []
        for i in range(len(insights)):
            if i in merged_indices: continue
            similar_indices = np.where(similarity_matrix[i] > self.dedup_threshold)[0]
            current_group_indices = [i] + [idx for idx in similar_indices if idx > i and idx not in merged_indices]
            best_insight_in_group = insights[i]; max_score = self._calculate_priority_score(best_insight_in_group)
            for group_idx in current_group_indices[1:]:
                 current_score = self._calculate_priority_score(insights[group_idx])
                 if current_score > max_score: max_score = current_score; best_insight_in_group = insights[group_idx]
                 merged_indices.add(group_idx)
            unique_insights_list.append(best_insight_in_group); merged_indices.add(i)
        return unique_insights_list

    def _cluster_insights(self, insights: List[StructuredInsight]) -> Tuple[List[List[StructuredInsight]], List[StructuredInsight]]:
        """Groups insights into clusters using OpenAI embeddings."""
        if len(insights) <= 1: return [], insights
        texts_to_embed = [self._get_insight_text(i) for i in insights]
        try:
             embeddings = self._get_embeddings(texts_to_embed) # Use helper method
        except Exception as e:
             logger.error(f"Failed to get embeddings for clustering: {e}. Skipping synthesis.")
             return [], insights # Return no clusters, all insights unclustered

        if embeddings.shape[0] != len(insights):
             logger.error("Embedding count mismatch. Skipping clustering.")
             return [], insights

        distance_threshold = 1.0 - self.synth_cluster_threshold
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric='cosine', linkage='average')
        try:
             labels = clustering.fit_predict(embeddings)
        except ValueError as e:
             logger.warning(f"Clustering failed ({e}), proceeding without synthesis.")
             return [], insights
        clusters = {}; unclustered_indices = []
        for i, label in enumerate(labels):
            if label < 0: unclustered_indices.append(i); continue
            if label not in clusters: clusters[label] = []
            clusters[label].append(i)
        final_clusters_indices: List[List[int]] = []
        for indices in clusters.values(): # No need for label key now
            if len(indices) > 1: final_clusters_indices.append(indices)
            else: unclustered_indices.extend(indices)
        clustered_insight_groups = [[insights[i] for i in index_group] for index_group in final_clusters_indices]
        unclustered_insight_list = [insights[i] for i in unclustered_indices]
        return clustered_insight_groups, unclustered_insight_list

    def _synthesize_clusters(self, clustered_insight_groups: List[List[StructuredInsight]], original_user_query: Optional[str], db_summary: Optional[DatabaseSummary]) -> List[StructuredInsight]:
        """Uses LLM to synthesize each cluster of insights."""
        synthesized_insights = []
        if not self.llm_client:
             logger.error("LLM client not available for synthesis.")
             return [] # Cannot synthesize without LLM client
        for i, cluster in enumerate(clustered_insight_groups):
            logger.info(f"Synthesizing cluster {i+1}/{len(clustered_insight_groups)}...")
            synthesis_prompt = self._build_synthesis_prompt(cluster, original_user_query, db_summary)
            try:
                synthesized_insight_data = self.llm_client.generate(
                    user_prompt=synthesis_prompt,
                    system_prompt="You are an expert data analyst synthesizing multiple related insights into a single, higher-level summary insight. Provide your response as a JSON object matching the fields of the StructuredInsight schema provided in the user prompt.",
                    response_model=StructuredInsight, temperature=0.2
                )
                if isinstance(synthesized_insight_data, StructuredInsight):
                     synthesized_insight_data.insight_id = f"SYN-{uuid.uuid4().hex[:10]}"
                     synthesized_insights.append(synthesized_insight_data)
                     logger.info(f"Successfully synthesized cluster {i+1}.")
                elif isinstance(synthesized_insight_data, str):
                     logger.warning("LLM returned string, attempting parse for synthesis.")
                     try:
                          parsed_data = json.loads(synthesized_insight_data); insight = StructuredInsight(**parsed_data)
                          insight.insight_id = f"SYN-{uuid.uuid4().hex[:10]}"; synthesized_insights.append(insight)
                          logger.info(f"Successfully synthesized cluster {i+1} (parsed string).")
                     except Exception as parse_error: logger.error(f"Failed to parse LLM string for synthesis: {parse_error}")
                else: logger.error(f"Unexpected LLM response type for synthesis: {type(synthesized_insight_data)}")
            except Exception as e: logger.error(f"LLM synthesis call failed cluster {i+1}: {e}", exc_info=True)
        return synthesized_insights

    def _build_synthesis_prompt(self, insights_to_synthesize: List[StructuredInsight], original_user_query: Optional[str], db_summary: Optional[DatabaseSummary]) -> str:
        """Constructs the prompt for the LLM to synthesize insights."""
        insights_text = ""; context = f"Original User Query: {original_user_query or 'N/A'}\n"
        for idx, insight in enumerate(insights_to_synthesize): insights_text += f"\n--- Insight {idx+1} (ID: {insight.insight_id}) ---\nHeadline: {insight.headline}\nDesc: {insight.description}\nTier: {insight.tier}\nMetrics: {json.dumps(insight.supporting_metrics)}\n"
        allowed_tiers = [t.value for t in InsightTier if t != InsightTier.EXECUTION_INFO]; tier_enum_str = ", ".join(allowed_tiers)
        prompt = f"""Synthesize the following related insights into a single, higher-level insight.\n\n**Context:**\n{context}\n**Insights to Synthesize:**{insights_text}\n**Your Task:**\nGenerate a *new* insight summarizing the overarching theme. Provide: `headline`, `description`, `tier` (one of: {tier_enum_str}), `supporting_metrics` (summarized), estimated scores (`relevance_score`, `significance_score`, `confidence_score`), `potential_actions`, `further_investigation_q`.\n**Format response strictly as a single JSON object matching the schema below.**\n\n**Required JSON Output Schema:**\n```json\n{{\n  "headline": "string", "description": "string", "tier": "string (Enum: {tier_enum_str})",\n  "relevance_score": "float | null", "significance_score": "float | null", "confidence_score": "float | null",\n  "supporting_metrics": {{...}}, "supporting_examples": null,\n  "comparison_details": "string | null", "trend_pattern": "string | null", "anomaly_description": "string | null", "contribution_details": "string | null",\n  "potential_actions": ["string", ...], "further_investigation_q": ["string", ...]\n}}\n```\nProvide ONLY the JSON object."""
        return prompt.strip()

    def _calculate_priority_score(self, insight: StructuredInsight) -> float:
        """Calculates a priority score based on configured weights."""
        score = 0.0; w = self.priority_weights; relevance = insight.relevance_score if insight.relevance_score is not None else 0.5; significance = insight.significance_score if insight.significance_score is not None else 0.5; confidence = insight.confidence_score if insight.confidence_score is not None else 0.5; tier_score = self.tier_scores.get(insight.tier, 0.1); score += relevance * w.get("relevance", 0.0) + significance * w.get("significance", 0.0) + confidence * w.get("confidence", 0.0) + tier_score * w.get("tier", 0.0); total_weight = sum(w.values()); score = score / total_weight if total_weight > 0 and total_weight != 1.0 else score; return score

    def _prioritize_insights(self, insights: List[StructuredInsight]) -> List[StructuredInsight]:
        """Sorts insights based on calculated priority score (highest first)."""
        if not insights:
            return []

        scored_insights = []
        for i, insight in enumerate(insights):
            try:
                score = self._calculate_priority_score(insight)
                scored_insights.append((score, insight))
            except Exception as e:
                logger.error(f"Failed to calculate priority score for insight {insight.insight_id}: {e}. Assigning score 0.", exc_info=True)
                # Assign a default low score or skip? Assigning 0 allows it to be sorted.
                scored_insights.append((0.0, insight))

        try:
            # Sort descending by score
            scored_insights.sort(key=lambda x: x[0], reverse=True)
            # Return only the insight objects in sorted order
            return [insight for score, insight in scored_insights]
        except Exception as e:
            logger.error(f"Failed to sort insights during prioritization: {e}", exc_info=True)
            # Fallback: return the original list (unsorted) if sorting fails
            return insights

    def _filter_insights(self, insights: List[StructuredInsight]) -> List[StructuredInsight]:
        """Filters insights, e.g., keeping Top-N."""
        if self.filter_top_n is None or self.filter_top_n <= 0: return insights; return insights[:self.filter_top_n]