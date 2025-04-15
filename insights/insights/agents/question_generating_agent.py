import logging
import json
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field
import string
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from insights.llm import call_openai_api
from insights.utils import setup_logging

logger = setup_logging()

# Set up NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK resource 'punkt' not found. Downloading...")
    nltk.download('punkt')

# >>> ADD THIS BLOCK <<<
try:
    # Explicitly check for punkt_tab as requested by the error
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("NLTK resource 'punkt_tab' not found. Downloading...")
    nltk.download('punkt_tab')
# >>> END OF ADDED BLOCK <<<

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("NLTK resource 'stopwords' not found. Downloading...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("NLTK resource 'wordnet' not found. Downloading...")
    nltk.download('wordnet')

class QuestionCategory(str, Enum):
    """Categories for analysis questions."""
    TREND_ANALYSIS = "trend analysis"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    ANOMALY_DETECTION = "anomaly detection"
    CORRELATION = "correlation" 
    DISTRIBUTION = "distribution"
    FORECASTING = "forecasting"

class AgentRole(str, Enum):
    """Roles for the LLM when generating content."""
    # Technical roles
    DATA_ANALYST = "data analyst"
    DATA_SCIENTIST = "data scientist"
    DATABASE_EXPERT = "database expert"
    
    # Business roles
    BUSINESS_ANALYST = "business analyst"
    MARKETING_SPECIALIST = "marketing specialist"
    FINANCIAL_ANALYST = "financial analyst"
    SALES_MANAGER = "sales manager"
    PRODUCT_MANAGER = "product manager"
    OPERATIONS_MANAGER = "operations manager"
    SUPPLY_CHAIN_ANALYST = "supply chain analyst"
    HR_ANALYST = "human resources analyst"
    CUSTOMER_SUCCESS_MANAGER = "customer success manager"
    
    # Executive roles
    CEO = "chief executive officer"
    CFO = "chief financial officer"
    CMO = "chief marketing officer"
    CTO = "chief technology officer"
    
    # Industry-specific roles
    RETAIL_ANALYST = "retail analyst"
    HEALTHCARE_ANALYST = "healthcare analyst"
    FINANCE_INDUSTRY_EXPERT = "finance industry expert"
    MANUFACTURING_SPECIALIST = "manufacturing specialist"
    E_COMMERCE_EXPERT = "e-commerce expert"

class RoleCategory(str, Enum):
    """Categories of agent roles."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    EXECUTIVE = "executive"
    INDUSTRY = "industry"

# Mapping of roles to their categories
ROLE_CATEGORY_MAP = {
    # Technical roles
    AgentRole.DATA_ANALYST: RoleCategory.TECHNICAL,
    AgentRole.DATA_SCIENTIST: RoleCategory.TECHNICAL,
    AgentRole.DATABASE_EXPERT: RoleCategory.TECHNICAL,
    
    # Business roles
    AgentRole.BUSINESS_ANALYST: RoleCategory.BUSINESS,
    AgentRole.MARKETING_SPECIALIST: RoleCategory.BUSINESS,
    AgentRole.FINANCIAL_ANALYST: RoleCategory.BUSINESS,
    AgentRole.SALES_MANAGER: RoleCategory.BUSINESS,
    AgentRole.PRODUCT_MANAGER: RoleCategory.BUSINESS,
    AgentRole.OPERATIONS_MANAGER: RoleCategory.BUSINESS,
    AgentRole.SUPPLY_CHAIN_ANALYST: RoleCategory.BUSINESS,
    AgentRole.HR_ANALYST: RoleCategory.BUSINESS,
    AgentRole.CUSTOMER_SUCCESS_MANAGER: RoleCategory.BUSINESS,
    
    # Executive roles
    AgentRole.CEO: RoleCategory.EXECUTIVE,
    AgentRole.CFO: RoleCategory.EXECUTIVE,
    AgentRole.CMO: RoleCategory.EXECUTIVE,
    AgentRole.CTO: RoleCategory.EXECUTIVE,
    
    # Industry-specific roles
    AgentRole.RETAIL_ANALYST: RoleCategory.INDUSTRY,
    AgentRole.HEALTHCARE_ANALYST: RoleCategory.INDUSTRY,
    AgentRole.FINANCE_INDUSTRY_EXPERT: RoleCategory.INDUSTRY,
    AgentRole.MANUFACTURING_SPECIALIST: RoleCategory.INDUSTRY,
    AgentRole.E_COMMERCE_EXPERT: RoleCategory.INDUSTRY,
}

# Moved outside the Enum class definition
ROLE_SPECIFIC_GUIDANCE = {
    # Technical roles
    "data analyst": "Focus on descriptive analytics and finding basic patterns in the data.",
    "data scientist": "Look for complex patterns, relationships, and opportunities for predictive modeling.",
    "database expert": "Consider database structure optimization and query performance aspects.",

    # Business roles
    "business analyst": "Emphasize questions that connect data patterns to business outcomes and decisions.",
    "marketing specialist": "Focus on customer acquisition, campaign effectiveness, and market segmentation.",
    "financial analyst": "Emphasize profitability, cost analysis, revenue patterns, and financial forecasting.",
    "sales manager": "Focus on sales performance, pipeline analysis, and customer conversion patterns.",
    "product manager": "Emphasize product performance, feature impact, and user behavior analytics.",
    "operations manager": "Focus on process efficiency, resource utilization, and operational bottlenecks.",
    "supply chain analyst": "Emphasize inventory patterns, logistics optimization, and supply chain risks.",
    "human resources analyst": "Focus on workforce analytics, employee performance, and talent management.",
    "customer success manager": "Emphasize customer satisfaction, retention patterns, and service improvement.",

    # Executive roles
    "chief executive officer": "Focus on high-level business performance, market positioning, and strategic decisions.",
    "chief financial officer": "Emphasize financial health, investment opportunities, and risk management.",
    "chief marketing officer": "Focus on brand performance, marketing ROI, and customer acquisition strategy.",
    "chief technology officer": "Emphasize technology performance, innovation opportunities, and technical debt.",

    # Industry-specific roles
    "retail analyst": "Focus on store performance, product mix analysis, and seasonal patterns.",
    "healthcare analyst": "Emphasize patient outcomes, care efficiency, and compliance metrics.",
    "finance industry expert": "Focus on risk profiles, investment performance, and market indicators.",
    "manufacturing specialist": "Emphasize production efficiency, quality metrics, and supply chain optimization.",
    "e-commerce expert": "Focus on conversion funnels, customer journey analysis, and digital marketing effectiveness."
}


class PromptTemplate(str, Enum):
    """Templates for system prompts."""
    QUESTION_GENERATION = """
    You are an expert {role} who specializes in formulating analytical questions.
    Your task is to generate insightful, specific analysis questions based on a user query and a database summary.

    As a {role}, focus on questions that would be most valuable from your professional perspective.

    For each question:
    1. Focus on generating questions that can be answered with SQL queries against the provided database
    2. Make questions specific and actionable (not vague or general)
    3. Ensure questions are relevant to the user's original query
    4. Tailor questions to the actual database structure provided
    5. Assign a relevance score (0.0-1.0) indicating how directly the question relates to the user query
    6. Categorize each question using one of the following categories: {categories}
    7. Include the specific tables and columns needed to answer each question

    Aim for a diverse set of questions that explore different aspects of the data from your perspective as a {role}.
    """

class AnalysisQuestion(BaseModel):
    """Model for a single analysis question."""
    question_id: str
    question_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    category: QuestionCategory
    related_tables: List[str] = Field(default_factory=list)
    related_columns: List[str] = Field(default_factory=list)
    source_role: Optional[str] = None  # Track which role generated this question
    processed_text: Optional[str] = None  # Pre-processed text for NLP similarity comparison

class AnalysisQuestionList(BaseModel):
    """Model for a list of analysis questions."""
    questions: List[AnalysisQuestion]

class GeneratedQuestions(BaseModel):
    """Model for the complete set of generated questions."""
    questions: List[AnalysisQuestion]
    total_questions: int
    roles_used: List[str] = Field(default_factory=list)
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    role_distribution: Dict[str, int] = Field(default_factory=dict)

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
        
        # Remove punctuation
        for char in string.punctuation:
            text = text.replace(char, ' ')
        
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

class QuestionGeneratingAgent:
    """
    Agent for generating analysis questions based on database summary and user query.
    
    This agent explores the database structure and suggests relevant questions
    that could lead to valuable insights from multiple professional perspectives.
    """
    
    def __init__(self, 
                 questions_per_role: int = 5,
                 similarity_threshold: float = 0.85,
                 min_relevance_score: float = 0.6,
                 num_roles: Optional[int] = None):
        """
        Initialize the Question Generating Agent.
        
        Args:
            questions_per_role: Number of questions to generate per role
            similarity_threshold: Threshold for considering questions as duplicates (0.0-1.0)
            min_relevance_score: Minimum relevance score for questions to be included
            num_roles: Number of different roles to use (None means use all roles)
        """
        self.questions_per_role = questions_per_role
        self.similarity_threshold = similarity_threshold
        self.min_relevance_score = min_relevance_score
        self.num_roles = num_roles if num_roles is not None and num_roles > 0 else 4
        self.text_processor = NLTKTextProcessor()
        
    def _build_system_prompt(self, role: AgentRole) -> str:
        """
        Build a system prompt from a template.

        Args:
            role: The role to use

        Returns:
            The formatted system prompt
        """
        categories_str = ", ".join([cat.value for cat in QuestionCategory])

        # Start with the base template
        prompt = PromptTemplate.QUESTION_GENERATION.value.format(
            role=role.value,
            categories=categories_str
        )

        # Add role-specific guidance if available
        # Access the module-level dictionary directly
        role_guidance = ROLE_SPECIFIC_GUIDANCE.get(role.value) # <-- Use the moved dictionary
        if role_guidance:
            prompt += f"\n\nAs a {role.value}, {role_guidance}"

        return prompt
        
    def _generate_question_id(self, role_prefix: str, index: int) -> str:
        """
        Generate a unique question ID.
        
        Args:
            role_prefix: Prefix indicating the source role
            index: The question number
            
        Returns:
            A formatted question ID (e.g., DA001 for Data Analyst question 1)
        """
        return f"{role_prefix}{str(index+1).zfill(3)}"
    
    def _get_role_prefix(self, role: AgentRole) -> str:
        """
        Get a short prefix for a role.
        
        Args:
            role: The agent role
            
        Returns:
            A 2-character prefix
        """
        role_prefixes = {
            # Technical roles
            AgentRole.DATA_ANALYST: "DA",
            AgentRole.DATA_SCIENTIST: "DS",
            AgentRole.DATABASE_EXPERT: "DE",
            
            # Business roles
            AgentRole.BUSINESS_ANALYST: "BA",
            AgentRole.MARKETING_SPECIALIST: "MS",
            AgentRole.FINANCIAL_ANALYST: "FA",
            AgentRole.SALES_MANAGER: "SM",
            AgentRole.PRODUCT_MANAGER: "PM",
            AgentRole.OPERATIONS_MANAGER: "OM",
            AgentRole.SUPPLY_CHAIN_ANALYST: "SC",
            AgentRole.HR_ANALYST: "HR",
            AgentRole.CUSTOMER_SUCCESS_MANAGER: "CS",
            
            # Executive roles
            AgentRole.CEO: "CE",
            AgentRole.CFO: "CF",
            AgentRole.CMO: "CM",
            AgentRole.CTO: "CT",
            
            # Industry-specific roles
            AgentRole.RETAIL_ANALYST: "RA",
            AgentRole.HEALTHCARE_ANALYST: "HA",
            AgentRole.FINANCE_INDUSTRY_EXPERT: "FI",
            AgentRole.MANUFACTURING_SPECIALIST: "MF",
            AgentRole.E_COMMERCE_EXPERT: "EC",
        }
        
        return role_prefixes.get(role, "QU")  # Default to "QU" (Question) if role not found
    
    def _select_diverse_roles(self) -> List[AgentRole]:
        """
        Select a diverse set of roles from different categories.
        
        Returns:
            A list of selected roles
        """
        # If using all roles
        if self.num_roles is None or self.num_roles >= len(AgentRole):
            return list(AgentRole)
            
        selected_roles = []
        
        # Group roles by category
        roles_by_category = {}
        for role, category in ROLE_CATEGORY_MAP.items():
            if category not in roles_by_category:
                roles_by_category[category] = []
            roles_by_category[category].append(role)
        
        # Try to select at least one role from each category
        for category in RoleCategory:
            if len(selected_roles) < self.num_roles and category in roles_by_category:
                roles = roles_by_category[category]
                if roles:
                    selected_roles.append(random.choice(roles))
        
        # If we still need more roles, randomly select from remaining roles
        if len(selected_roles) < self.num_roles:
            all_roles = list(AgentRole)
            remaining_roles = [r for r in all_roles if r not in selected_roles]
            additional_roles = random.sample(
                remaining_roles, 
                min(self.num_roles - len(selected_roles), len(remaining_roles))
            )
            selected_roles.extend(additional_roles)
            
        return selected_roles
    
    def _generate_questions_for_role(self, 
                                    role: AgentRole, 
                                    user_query: str, 
                                    db_summary: Dict[str, Any]) -> List[AnalysisQuestion]:
        """
        Generate questions using a specific role.
        
        Args:
            role: The role to use
            user_query: The user's query
            db_summary: The database summary
            
        Returns:
            A list of questions
        """
        # Build the system prompt
        system_prompt = self._build_system_prompt(role)
        
        # Get role prefix for IDs
        role_prefix = self._get_role_prefix(role)

        # Categories
        categories = ", ".join([cat.value for cat in QuestionCategory])
        
        # Build the user prompt
        user_prompt = f"""
        USER QUERY: "{user_query}"
        
        DATABASE SUMMARY:
        {json.dumps(db_summary, indent=2)}
        
        Based on the above information, generate exactly {self.questions_per_role} different analytical questions
        from your perspective as a {role.value}.
        
        For each question, consider both:
        - What would directly answer the user's query
        - What additional insights might be valuable related to the query
        - What specific questions someone in your role would typically ask
        
        Format each question as a JSON object with the following structure:
        {{
            "question_id": "Q001",  // Will be replaced with a unique ID
            "question_text": "The specific analytical question text",
            "relevance_score": 0.85,  // A number between 0 and 1 indicating relevance to the original query
            "category": "One of: {categories}",
            "related_tables": ["specific tables needed to answer this question"],
            "related_columns": ["specific columns needed to answer this question"]
        }}
        
        Return a JSON array of these question objects. It should be called "questions". This would be the only key in the JSON object.
        """
        
        # Call the LLM to generate questions
        result = call_openai_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=AnalysisQuestionList,
            temperature=0.8  
        )
        
        if not result or not isinstance(result, AnalysisQuestionList):
            logger.error(f"Failed to generate questions for role: {role.value}")
            return []
        
        # Process and enrich the questions
        for i, question in enumerate(result.questions):
            # Update question_id
            question.question_id = self._generate_question_id(role_prefix, i)
            
            # Add source role
            question.source_role = role.value
            
            # Add processed text for NLP similarity comparison
            question.processed_text = self.text_processor.preprocess_text(question.question_text)
        
        # Filter by relevance score
        filtered_questions = [q for q in result.questions if q.relevance_score >= self.min_relevance_score]
        
        return filtered_questions
    
    def _deduplicate_questions_nltk(self, questions: List[AnalysisQuestion]) -> List[AnalysisQuestion]:
        """
        Remove duplicate or very similar questions using NLTK.
        
        Args:
            questions: List of questions
            
        Returns:
            Deduplicated list of questions
        """
        if not questions:
            return []
            
        # Sort by relevance score so we keep the most relevant version of similar questions
        sorted_questions = sorted(questions, key=lambda q: q.relevance_score, reverse=True)
        
        # Extract processed texts
        processed_texts = [q.processed_text for q in sorted_questions if q.processed_text]
        
        # If no processed texts, return all questions
        if not processed_texts:
            return sorted_questions
            
        # Compute similarity matrix
        similarity_matrix = self.text_processor.compute_similarity(processed_texts)
        
        unique_questions = []
        used_indices = set()
        
        # Iterate through questions in order of relevance
        for i, question in enumerate(sorted_questions):
            # Skip if this index has already been used as a duplicate
            if i in used_indices:
                continue
                
            # Add this question to unique questions
            unique_questions.append(question)
            
            # Find all questions that are similar to this one
            for j in range(i + 1, len(sorted_questions)):
                # Skip if already used
                if j in used_indices:
                    continue
                    
                # Check similarity
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    used_indices.add(j)
                    
        return unique_questions
    
    def generate_questions(self, user_query: str, db_summary: Dict[str, Any]) -> GeneratedQuestions:
        """
        Generate analysis questions from multiple perspectives.
        
        Args:
            user_query: The user's query
            db_summary: The database summary
            
        Returns:
            A structure containing generated questions and metadata
        """
        # Select diverse roles
        selected_roles = self._select_diverse_roles()
        logger.info(f"Selected roles: {[role.value for role in selected_roles]}")
        
        all_questions = []
        roles_used = []
        
        # Generate questions for each selected role
        for role in selected_roles:
            questions = self._generate_questions_for_role(role, user_query, db_summary)
            
            if questions:
                all_questions.extend(questions)
                roles_used.append(role.value)
                logger.info(f"Generated {len(questions)} questions from {role.value}")
            else:
                logger.warning(f"No questions generated from {role.value}")
        
        # Deduplicate questions using NLTK
        unique_questions = self._deduplicate_questions_nltk(all_questions)
        logger.info(f"After deduplication: {len(unique_questions)} unique questions from {len(all_questions)} total")
        
        # Calculate category distribution
        category_distribution = {}
        for question in unique_questions:
            category = question.category
            if isinstance(category, QuestionCategory):
                category = category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # Calculate role distribution
        role_distribution = {}
        for question in unique_questions:
            role = question.source_role
            if role:
                role_distribution[role] = role_distribution.get(role, 0) + 1
        
        return GeneratedQuestions(
            questions=unique_questions,
            total_questions=len(unique_questions),
            roles_used=roles_used,
            category_distribution=category_distribution,
            role_distribution=role_distribution
        )
    
    def generate_questions_with_roles(self, 
                                     user_query: str, 
                                     db_summary: Dict[str, Any],
                                     roles: List[AgentRole]) -> GeneratedQuestions:
        """
        Generate analysis questions using specific roles.
        
        Args:
            user_query: The user's query
            db_summary: The database summary
            roles: The roles to use
            
        Returns:
            A structure containing generated questions and metadata
        """
        all_questions = []
        roles_used = []
        
        # Generate questions for each role
        for role in roles:
            questions = self._generate_questions_for_role(role, user_query, db_summary)
            
            if questions:
                all_questions.extend(questions)
                roles_used.append(role.value)
                logger.info(f"Generated {len(questions)} questions from {role.value}")
            else:
                logger.warning(f"No questions generated from {role.value}")
        
        # Deduplicate questions using NLTK
        unique_questions = self._deduplicate_questions_nltk(all_questions)
        logger.info(f"After deduplication: {len(unique_questions)} unique questions from {len(all_questions)} total")
        
        # Calculate distributions
        category_distribution = {}
        role_distribution = {}
        
        for question in unique_questions:
            # Category distribution
            category = question.category
            if isinstance(category, QuestionCategory):
                category = category.value
            category_distribution[category] = category_distribution.get(category, 0) + 1
            
            # Role distribution
            role = question.source_role
            if role:
                role_distribution[role] = role_distribution.get(role, 0) + 1
        
        return GeneratedQuestions(
            questions=unique_questions,
            total_questions=len(unique_questions),
            roles_used=roles_used,
            category_distribution=category_distribution,
            role_distribution=role_distribution
        )

def parse_roles(role_strings: List[str]) -> List[AgentRole]:
    """
    Parse role strings into AgentRole enum values.
    
    Args:
        role_strings: List of role strings
        
    Returns:
        List of AgentRole values
    """
    # Check if "all" is specified
    if len(role_strings) == 1 and role_strings[0].lower() == "all":
        return list(AgentRole)
        
    roles = []
    
    for role_str in role_strings:
        try:
            # Try exact match first
            role = AgentRole(role_str)
            roles.append(role)
        except ValueError:
            # If not found, try a more flexible approach
            role_str_lower = role_str.lower()
            found = False
            
            for role in AgentRole:
                if role_str_lower in role.value:
                    roles.append(role)
                    found = True
                    break
            
            if not found:
                logger.warning(f"Could not find matching role for '{role_str}'")
    
    return roles

def main(user_query: str, db_summary_path: str, role_strings: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Entry point function to run the Question Generating Agent.
    
    Args:
        user_query: The user's query
        db_summary_path: Path to the database summary file
        role_strings: Optional list of role strings
        
    Returns:
        Dictionary with generated questions and metadata
    """
    # Load database summary
    with open(db_summary_path, 'r') as f:
        db_summary = json.load(f)
    
    # Create question generating agent
    agent = QuestionGeneratingAgent()
    
    # Generate questions
    if role_strings:
        # Parse roles
        roles = parse_roles(role_strings)
        if not roles:
            logger.warning("No valid roles provided, using automatic role selection")
            result = agent.generate_questions(user_query, db_summary)
        else:
            result = agent.generate_questions_with_roles(user_query, db_summary, roles)
    else:
        # Use automatic role selection
        result = agent.generate_questions(user_query, db_summary)
    
    return result.model_dump()

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python question_generating_agent.py <user_query> <path_to_db_summary.json> [role1 role2 ... | all]")
        sys.exit(1)
    
    user_query = sys.argv[1]
    db_summary_path = sys.argv[2]
    role_strings = sys.argv[3:] if len(sys.argv) > 3 else None
    
    # Support for "all" as a special argument
    if role_strings and len(role_strings) == 1 and role_strings[0].lower() == "all":
        logger.info("Using all available roles")
        role_strings = [role.value for role in AgentRole]
    
    result = main(user_query, db_summary_path, role_strings)
    print(json.dumps(result, indent=2))

    with open("questions.json", "w") as f:
        json.dump(result, f, indent=2)