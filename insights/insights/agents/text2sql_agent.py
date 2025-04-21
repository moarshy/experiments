import json
import time
import pandas as pd
from typing import Dict, Any, List, Optional
import concurrent.futures

from pydantic import BaseModel, Field

# Import Vanna components
from vanna.openai import OpenAI_Chat
from vanna.google import GoogleGeminiChat
from vanna.chromadb import ChromaDB_VectorStore

# Import from your existing modules
# Assuming DatabaseSummary structure is defined elsewhere
from insights.agents.db_summary_agent import DatabaseSummary, TableSummary, ColumnSummary
from insights.utils import setup_logging
from insights.config import OPENAI_API_KEY, DB_CONFIG, SQL_LLM_PROVIDER, DEFAULT_GEMINI_MODEL, GOOGLE_API_KEY

logger = setup_logging()

# --- Pydantic Models ---
class SQLResult(BaseModel):
    """Model for structured SQL query execution results."""
    query: str
    success: bool
    error_message: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int] = None
    column_names: Optional[List[str]] = None
    execution_time: Optional[float] = None

class ProcessedQuestionResult(BaseModel):
    """Combines input question data with SQL generation and execution results."""
    # Input Question Data (adjust fields based on AnalysisQuestion model)
    question_id: str
    question_text: str
    source_llm: Optional[str] = None
    iteration_level: Optional[int] = None
    # Add any other relevant fields from your AnalysisQuestion model
    category: Optional[str] = None
    relevance_score: Optional[float] = None

    # SQL Generation & Execution Data
    generated_sql: Optional[str] = None
    execution_success: bool = False # Default to False
    execution_error_message: Optional[str] = None
    execution_data: Optional[List[Dict[str, Any]]] = None
    execution_row_count: Optional[int] = None
    execution_column_names: Optional[List[str]] = None
    execution_time: Optional[float] = None
    processing_error: Optional[str] = None # For errors before/during SQL gen

# --- Vanna Agent Combination ---

class VannaAgentOpenAI(ChromaDB_VectorStore, OpenAI_Chat):
    """Custom Vanna agent combining ChromaDB VectorStore and OpenAI Chat capabilities."""
    def __init__(self, config=None):
        # Ensure ChromaDB is initialized first if OpenAI_Chat depends on it indirectly
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)
        logger.info("Initialized VannaAgent with ChromaDB and OpenAI")

class VannaAgentGemini(ChromaDB_VectorStore, GoogleGeminiChat):
    """Custom Vanna agent combining ChromaDB VectorStore and Google Gemini Chat capabilities."""
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        GoogleGeminiChat.__init__(self, config=config)
        logger.info("Initialized VannaAgent with ChromaDB and Google Gemini")

# --- Core Agent Class ---

class Text2SQLExecuteAgent:
    """
    Generates and executes SQL from natural language using Vanna AI.

    Connects to a database, optionally trains on its schema/summary,
    translates questions to SQL, executes them, and returns structured results.
    """

    def __init__(self,
                 openai_model: str = "gpt-4.1",
                 collection_name: str = "insights_db", # Use a descriptive name
                 persist_directory: str = "./vanna_data"):
        """
        Initializes the agent.

        Args:
            openai_model: The OpenAI model identifier to use with Vanna.
            collection_name: Name for the Vanna ChromaDB collection.
            persist_directory: Local directory to persist ChromaDB data.
        """
        self.db_config = DB_CONFIG
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not configured.")
        if not self.db_config:
             raise ValueError("DB_CONFIG is not configured.")

        self.vanna_config = {
            'api_key': OPENAI_API_KEY if SQL_LLM_PROVIDER == 'openai' else GOOGLE_API_KEY,
            'model': openai_model if SQL_LLM_PROVIDER == 'openai' else DEFAULT_GEMINI_MODEL,
            'collection_name': collection_name,
            'path': persist_directory
        }

        try:
            if SQL_LLM_PROVIDER == 'openai':
                self.vn = VannaAgentOpenAI(config=self.vanna_config)
            elif SQL_LLM_PROVIDER == 'gemini':
                self.vn = VannaAgentGemini(config=self.vanna_config)
        except Exception as e:
            logger.error(f"Failed to initialize VannaAgent: {e}", exc_info=True)
            raise

        self.is_connected = False
        self.is_trained = False
        logger.info(f"Text2SQLExecuteAgent initialized with model '{openai_model}' and collection '{collection_name}'.")

    def connect_to_database(self) -> bool:
        """Connects Vanna to the configured database."""
        if self.is_connected:
            logger.info("Already connected to the database.")
            return True

        db_type = self.db_config.get('type', 'sqlite').lower()
        logger.info(f"Attempting to connect to {db_type} database...")

        try:
            if db_type == 'sqlite':
                db_path = self.db_config.get('database')
                if not db_path: raise ValueError("SQLite 'database' path not configured in DB_CONFIG.")
                self.vn.connect_to_sqlite(db_path)

            elif db_type == 'postgres':
                self.vn.connect_to_postgres(
                    host=self.db_config.get('host', 'localhost'),
                    dbname=self.db_config.get('dbname'),
                    user=self.db_config.get('user'),
                    password=self.db_config.get('password'),
                    port=self.db_config.get('port', '5432')
                )
            # Add elif blocks for other database types supported by Vanna if needed
            # elif db_type == 'mysql': self.vn.connect_to_mysql(...)
            # elif db_type == 'bigquery': self.vn.connect_to_bigquery(...)
            else:
                logger.error(f"Unsupported database type '{db_type}' configured.")
                return False

            self.is_connected = True
            logger.info(f"Successfully connected to {db_type} database.")
            return True

        except Exception as e:
            logger.error(f"Database connection failed: {e}", exc_info=True)
            self.is_connected = False
            return False

    def train_on_database_summary(self, db_summary: DatabaseSummary) -> bool:
        """
        Trains Vanna using schema information derived from the DatabaseSummary.

        Generates DDL, adds table/column descriptions as documentation.

        Args:
            db_summary: A DatabaseSummary object containing schema info.

        Returns:
            True if training was attempted (check logs for details), False if connection failed.
        """
        if not self.is_connected:
            logger.warning("Not connected to database. Attempting connection before training.")
            if not self.connect_to_database():
                logger.error("Cannot train: Database connection failed.")
                return False

        logger.info("Starting Vanna training based on Database Summary...")
        try:
            # Train on DDL for each table
            for table in db_summary.tables:
                ddl = self._create_ddl_from_table_summary(table)
                if ddl:
                    logger.debug(f"Training DDL for table {table.name}:\n{ddl}")
                    self.vn.train(ddl=ddl)

                # Train on table-level documentation
                table_doc = f"Table '{table.name}' contains {table.row_count} rows."
                if hasattr(table, 'description') and table.description:
                    table_doc += f" Description: {table.description}"
                self.vn.train(documentation=table_doc)

                # Train on column-level documentation
                for column in table.columns:
                    col_doc = f"Column '{column.name}' in table '{table.name}' has data type {column.data_type}."
                    if column.is_primary_key: col_doc += " It is a PRIMARY KEY."
                    if column.is_foreign_key and column.references: col_doc += f" It is a FOREIGN KEY referencing {column.references}."
                    if not column.nullable: col_doc += " It is NOT NULL."
                    if hasattr(column, 'description') and column.description: col_doc += f" Description: {column.description}"
                    self.vn.train(documentation=col_doc)

            # Train on relationships
            for relationship in db_summary.relationships:
                rel_doc = (f"Relationship: Column '{relationship.get('from_column')}' in table '{relationship.get('from_table')}' "
                           f"references column '{relationship.get('to_column')}' in table '{relationship.get('to_table')}'.")
                self.vn.train(documentation=rel_doc)

            # Optional: Attempt to train on INFORMATION_SCHEMA if available
            # This provides more comprehensive metadata directly from the DB
            try:
                logger.info("Attempting additional training using INFORMATION_SCHEMA...")
                df_info_schema = self.vn.run_sql("SELECT table_schema, table_name, column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS")
                # Vanna's generic plan might work, or you might need specific logic per DB type
                plan = self.vn.get_training_plan_generic(df_info_schema)
                logger.debug(f"Generated training plan from INFORMATION_SCHEMA: {plan}")
                self.vn.train(plan=plan)
                logger.info("Successfully added training from INFORMATION_SCHEMA.")
            except Exception as e:
                # This is often expected if the DB doesn't have INFORMATION_SCHEMA or lacks permissions
                logger.warning(f"Could not train on INFORMATION_SCHEMA (this might be expected): {e}")

            self.is_trained = True
            logger.info("Vanna training based on Database Summary completed.")
            return True

        except Exception as e:
            logger.error(f"Error during Vanna training: {e}", exc_info=True)
            return False # Indicate training attempt failed

    def _create_ddl_from_table_summary(self, table_summary: TableSummary) -> Optional[str]:
        """Helper to generate a basic CREATE TABLE DDL statement from summary data."""
        if not table_summary or not table_summary.name or not table_summary.columns:
            logger.warning(f"Skipping DDL generation for incomplete table summary: {table_summary}")
            return None

        col_defs = []
        for col in table_summary.columns:
            col_str = f"  `{col.name}` {col.data_type}" # Use backticks for potential reserved words
            if col.is_primary_key:
                 # Note: Composite PKs need different handling; this assumes single col PKs from summary flag
                 # Real DDL might list PKs at the end. Vanna might handle this.
                 col_str += " PRIMARY KEY"
            if not col.nullable:
                 col_str += " NOT NULL"
            # Add other constraints like UNIQUE if available in summary
            col_defs.append(col_str)

        # Add multi-column primary keys if listed separately
        # if table_summary.primary_keys:
        #     pk_cols = ', '.join([f"`{pk}`" for pk in table_summary.primary_keys])
        #     col_defs.append(f"  PRIMARY KEY ({pk_cols})")

        # Add foreign keys
        for fk in table_summary.foreign_keys:
            if 'column' in fk and 'references' in fk and '.' in fk['references']:
                 ref_table, ref_col = fk['references'].split('.', 1)
                 col_defs.append(f"  FOREIGN KEY (`{fk['column']}`) REFERENCES `{ref_table}` (`{ref_col}`)")

        if not col_defs:
            logger.warning(f"No column definitions generated for table '{table_summary.name}'.")
            return None

        ddl = f"CREATE TABLE `{table_summary.name}` (\n"
        ddl += ",\n".join(col_defs)
        ddl += "\n);"
        return ddl

    def generate_sql(self, question: str, db_context: Optional[str] = None) -> str:
        """
        Generates SQL from a natural language question using Vanna.

        Args:
            question: The natural language question.
            db_context: Optional additional context string to provide to Vanna.

        Returns:
            The generated SQL query string.

        Raises:
            ConnectionError: If not connected to the database.
            RuntimeError: If Vanna fails to generate SQL.
        """
        if not self.is_connected:
            raise ConnectionError("Cannot generate SQL: Not connected to database.")
        if not self.is_trained:
            logger.warning("Vanna may produce less accurate SQL as training was not completed or confirmed.")

        # Enhance question with context if provided
        # Vanna's generate_sql might handle context implicitly via training,
        # but explicit context can sometimes help complex queries.
        # Consider if Vanna's specific implementation prefers context in the question vs. training data.
        prompt_question = question
        if db_context:
             prompt_question = f"Context: {db_context}\n\nQuestion: {question}"
             logger.debug("Using enhanced question with context for SQL generation.")

        try:
            logger.info(f"Generating SQL for question: '{question}'")
            sql = self.vn.generate_sql(question=prompt_question) # Use the potentially enhanced question
            if not sql:
                 raise RuntimeError("Vanna returned an empty SQL query.")
            logger.info(f"Generated SQL: {sql}")
            return sql
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}", exc_info=True)
            raise RuntimeError(f"Vanna SQL generation failed: {e}") from e

    def execute_sql(self, sql: str) -> SQLResult:
        """
        Executes a SQL query using Vanna and returns structured results.

        Args:
            sql: The SQL query string to execute.

        Returns:
            An SQLResult object containing execution details and data/error.
        """
        if not self.is_connected:
            logger.error("Cannot execute SQL: Not connected to database.")
            return SQLResult(query=sql, success=False, error_message="Not connected to database")

        logger.info(f"Executing SQL: {sql}")
        start_time = time.time()
        try:
            # vn.run_sql typically returns a pandas DataFrame for SELECTs,
            # and might return None or raise an error for other DML/DDL.
            # We assume it executes non-SELECTs if it doesn't raise an error.
            result_data = self.vn.run_sql(sql)
            end_time = time.time()
            exec_time = end_time - start_time

            if isinstance(result_data, pd.DataFrame):
                logger.info(f"SQL execution successful. Retrieved {len(result_data)} rows in {exec_time:.4f}s.")
                data_list = result_data.to_dict(orient='records')
                return SQLResult(
                    query=sql,
                    success=True,
                    data=data_list,
                    row_count=len(data_list),
                    column_names=list(result_data.columns),
                    execution_time=exec_time
                )
            else:
                # Assume success for non-SELECT if no error occurred
                logger.info(f"SQL executed successfully (non-SELECT or empty result) in {exec_time:.4f}s.")
                return SQLResult(
                    query=sql,
                    success=True,
                    data=[], # Empty list for consistency
                    row_count=0, # Or None if preferred for non-SELECT
                    execution_time=exec_time
                )

        except Exception as e:
            end_time = time.time()
            exec_time = end_time - start_time
            logger.error(f"SQL execution failed after {exec_time:.4f}s: {e}", exc_info=True)
            return SQLResult(
                query=sql,
                success=False,
                error_message=str(e),
                execution_time=exec_time # Still record time if possible
            )

    def _process_single_question(self, q_data: Dict[str, Any], db_context: Optional[str]) -> ProcessedQuestionResult:
        """Processes a single question end-to-end."""
        question_text = q_data.get("question_text")
        if not question_text:
            # This case should ideally be filtered out before submitting to the pool
            # but handle defensively here too.
            logger.warning(f"Processing skipped for item with missing 'question_text': {q_data.get('question_id')}")
            return ProcessedQuestionResult(
                **q_data,
                execution_success=False,
                processing_error="Missing 'question_text' in input."
            )

        generated_sql: Optional[str] = None
        try:
            # Generate SQL
            # Use the potentially richer specific_context if available, otherwise db_context
            specific_context = db_context # In parallel version, complex context per question is harder
                                          # Keep it simple for now, pass the general context.
            generated_sql = self.generate_sql(question_text, specific_context)

            # Execute SQL
            exec_result = self.execute_sql(generated_sql)

            # Combine results using original q_data
            return ProcessedQuestionResult(
                **q_data, # Unpack original question data
                generated_sql=generated_sql,
                execution_success=exec_result.success,
                execution_error_message=exec_result.error_message,
                execution_data=exec_result.data,
                execution_row_count=exec_result.row_count,
                execution_column_names=exec_result.column_names,
                execution_time=exec_result.execution_time
            )
        except (ConnectionError, RuntimeError, Exception) as e:
            # Catch errors during SQL generation or execution for this specific question
            logger.error(f"Failed to fully process question ID {q_data.get('question_id')}: {e}", exc_info=True)
            return ProcessedQuestionResult(
                **q_data,
                generated_sql=generated_sql, # Include SQL if generation succeeded before error
                execution_success=False,
                processing_error=f"Error during processing: {e}"
            )

    def process_analysis_questions(self,
                                  analysis_questions: List[Dict[str, Any]],
                                  db_summary: Optional[DatabaseSummary] = None,
                                  max_workers: Optional[int] = None) -> List[ProcessedQuestionResult]:
        """
        Processes a list of analysis questions end-to-end in parallel.

        Generates SQL, executes it, and combines results with original question data.

        Args:
            analysis_questions: A list of dictionaries, each representing an analysis question
                                (should contain keys matching ProcessedQuestionResult input fields).
            db_summary: Optional DatabaseSummary object to extract context from.
            max_workers: Optional maximum number of threads to use. Defaults to None (Python's default).

        Returns:
            A list of ProcessedQuestionResult objects.
        """
        results: List[ProcessedQuestionResult] = []
        db_context = db_summary.natural_language_summary if db_summary and hasattr(db_summary, 'natural_language_summary') else ""

        if not self.is_connected:
            logger.error("Cannot process questions: Not connected to the database.")
            # Return partial results indicating connection failure for all questions
            for q_data in analysis_questions:
                 results.append(ProcessedQuestionResult(
                    **q_data,
                    execution_success=False,
                    processing_error="Database connection failed before processing."
                 ))
            return results

        # Filter out invalid questions upfront
        valid_questions_data = []
        invalid_questions_results = []
        for q_data in analysis_questions:
            if q_data.get("question_text"):
                valid_questions_data.append(q_data)
            else:
                logger.warning(f"Skipping question data with missing 'question_text': {q_data.get('question_id')}")
                invalid_questions_results.append(ProcessedQuestionResult(
                    **q_data,
                    execution_success=False,
                    processing_error="Missing 'question_text' in input."
                ))

        # Use ThreadPoolExecutor for parallel processing
        results_map = {} # To map future back to original index if order matters, though list append below is fine
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_qdata = {
                executor.submit(self._process_single_question, q_data, db_context): q_data
                for q_data in valid_questions_data
            }

            logger.info(f"Submitted {len(future_to_qdata)} questions for processing...")

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_qdata):
                q_data_orig = future_to_qdata[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    # Catch exceptions raised *by the future itself* (less likely with try/except in _process_single_question)
                    logger.error(f"Question ID {q_data_orig.get('question_id')} generated an unexpected exception: {exc}", exc_info=True)
                    results.append(ProcessedQuestionResult(
                        **q_data_orig,
                        execution_success=False,
                        processing_error=f"Unexpected error during future execution: {exc}"
                    ))

        # Add back results for invalid questions
        results.extend(invalid_questions_results)

        logger.info(f"Finished processing {len(analysis_questions)} initial questions ({len(results)} results generated).")
        # Note: Results list might not be in the original order of analysis_questions
        # If order is critical, use the results_map approach mentioned above or sort afterwards.
        return results

# --- Main Execution / Example Usage ---

def main(db_summary_path: str, questions_path: str, results_output_path: str = "processed_results.json"):
    """
    Main function to run the agent processing flow.

    Args:
        db_summary_path: Path to the JSON file containing the DatabaseSummary.
        questions_path: Path to the JSON file containing the list of analysis questions.
        results_output_path: Path to save the JSON output of processed results.
    """
    logger.info("--- Starting Text2SQL Execution Pipeline ---")

    # Load Database Summary
    try:
        with open(db_summary_path, 'r', encoding='utf-8') as f:
            # Load and parse summary into the Pydantic model for validation
            db_summary_data = json.load(f)
            # Assuming the summary file contains the structure matching DatabaseSummary model
            # If it's nested (e.g., under a 'technical_summary' key), adjust accordingly:
            if "technical_summary" in db_summary_data:
                 db_summary = DatabaseSummary(**db_summary_data["technical_summary"])
                 # Optionally add natural language summary if it's separate
                 if "natural_language_summary" in db_summary_data:
                      db_summary.natural_language_summary = db_summary_data["natural_language_summary"]
            else:
                 db_summary = DatabaseSummary(**db_summary_data)

            logger.info(f"Successfully loaded database summary from {db_summary_path}")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to load or parse database summary from {db_summary_path}: {e}", exc_info=True)
        return # Cannot proceed without summary for training context

    # Load Analysis Questions
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions_input = json.load(f)
            # Assuming questions are in a list under a key like "questions"
            analysis_questions = questions_input.get("questions", [])
            if not analysis_questions:
                 logger.warning(f"No questions found in {questions_path} under the 'questions' key.")
                 return
            logger.info(f"Successfully loaded {len(analysis_questions)} questions from {questions_path}")
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to load or parse questions from {questions_path}: {e}", exc_info=True)
        return

    # Initialize Agent
    try:
        agent = Text2SQLExecuteAgent()
    except (ValueError, Exception) as e:
        logger.error(f"Failed to initialize Text2SQLExecuteAgent: {e}", exc_info=True)
        return

    # Connect and Train (Training is optional but recommended)
    if agent.connect_to_database():
        agent.train_on_database_summary(db_summary) # Train using the loaded summary
    else:
        logger.error("Proceeding without database connection. SQL execution will fail.")
        # Decide if you want to exit or continue without execution capability
        # return

    # Process Questions
    processed_results = agent.process_analysis_questions(analysis_questions, db_summary)

    # Save Results
    try:
        # Convert Pydantic models to dictionaries for JSON serialization
        results_to_save = [result.model_dump(exclude_none=True) for result in processed_results]
        with open(results_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved {len(processed_results)} processed results to {results_output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {results_output_path}: {e}", exc_info=True)

    logger.info("--- Text2SQL Execution Pipeline Finished ---")


if __name__ == "__main__":
    import sys
    import os

    # Basic argument check
    if len(sys.argv) != 3:
        print("Usage: python text2sql_execute_agent.py <path_to_db_summary.json> <path_to_questions.json>")
        print("Example: python text2sql_execute_agent.py ./database_summary.json ./generated_questions.json")
        sys.exit(1)

    db_summary_file = sys.argv[1]
    questions_file = sys.argv[2]
    results_file = "processed_results.json" # Default output file name

    # Check if input files exist
    if not os.path.exists(db_summary_file):
         print(f"Error: Database summary file not found at '{db_summary_file}'")
         sys.exit(1)
    if not os.path.exists(questions_file):
         print(f"Error: Questions file not found at '{questions_file}'")
         sys.exit(1)

    # Run the main processing function
    main(db_summary_path=db_summary_file,
         questions_path=questions_file,
         results_output_path=results_file)