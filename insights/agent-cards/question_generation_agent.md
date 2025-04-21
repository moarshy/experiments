Agent Card: Iterative Business Question Generator (v1.1)
1. Agent Name:
Iterative Business Question Generator

2. Version:
v1.1 (Refined configuration from v1)

3. Core Objective:
To autonomously generate a set of meaningful, complex, and contextually relevant analytical questions about a dataset, primarily focused on uncovering business insights. It aims to move beyond basic queries by progressively building upon previously generated questions, potentially leveraging multiple LLM perspectives within the iterative process.

4. Key Features & Capabilities:

Iterative Question Generation: Employs a loop-based approach where LLMs are prompted multiple times. Each subsequent prompt includes previously generated questions as context, pushing the LLMs to generate more complex, follow-up, or deeper-dive questions. The process continues for a configured number of iterations for each specified model.
Business-Centric Focus (Default): By default, prioritizes generating questions relevant to business strategy, performance analysis, customer behavior, market trends, financial health, and operational efficiency.
Meaningful & Complex Questions: Leverages advanced LLM reasoning capabilities to synthesize questions that require non-trivial analysis, correlation, or comparison.
Optional User Query: Can operate in two modes:
Exploratory Mode: Generates questions based only on the database summary.
Guided Mode: Uses an optional initial user query as a starting seed or thematic guide.
Multi-LLM Iteration: Utilizes a provided list of LLMs. Each model in the list performs a specified number of generation iterations, contributing questions to the overall pool. This allows leveraging the potential strengths of different models.
Granular Generation Control: Allows precise control over the generation process via max_iterations_per_model and num_questions_per_iteration_per_model parameters.
Simplified Processing: Removes the dependency on NLTK for text processing and deduplication, relying on the iterative process and potential downstream consolidation.
5. Inputs:

Database Summary: A structured representation (e.g., JSON from the DatabaseSummaryAgent) or a detailed natural language summary of the database.
(Optional) User Query: A natural language question or topic.
Configuration:
llm_list: List[str]: A list of LLM identifiers to use (e.g., ['gpt-4-turbo', 'gemini-1.5-pro']).
max_iterations_per_model: int: The maximum number of iterative generation cycles each model in llm_list will perform.
num_questions_per_iteration_per_model: int: The target number of new questions each model should attempt to generate in each of its iterations.
(Future) Desired Question Themes (e.g., business, technical, data_quality)
6. Outputs:

GeneratedQuestions Object: Containing:
A list of AnalysisQuestion objects (potentially including source LLM and iteration level).
Metadata: Models used, total iterations performed across all models, final question count (pre-consolidation).
7. Key Design Principles & Changes:

Depth via Iteration & Multi-Model Input: Achieves question depth and diversity through controlled iteration across potentially multiple LLMs.
Configurable Exploration: Provides fine-grained control over the computational effort and the volume/depth of question generation.
Business-Centric Default: Prioritizes questions relevant to business users.
Decoupling from Roles: Moves away from explicit multi-role personas.
Reduced Pre-processing: Simplifies the agent's internal logic.
8. Potential Future Enhancements:

More sophisticated iteration strategies (e.g., adaptive selection of the next model based on previous output, branching logic).
Integration of semantic similarity checks/consolidation within or after the generation loop.
Dynamic adjustment of prompts based on intermediate results.
Configurable question themes.
