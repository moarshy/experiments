import logging
import sqlite3
import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from insights.llm import gemini_call, openai_call
from insights.utils import setup_logging

logger = setup_logging()

class TimeRangeValue(BaseModel):
    """Represents the min and max values for a time range."""
    min: Optional[str] = None
    max: Optional[str] = None

class ColumnSummary(BaseModel):
    """Summary information for a database column."""
    name: str
    data_type: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references: Optional[str] = None
    nullable: bool = True
    unique: bool = False
    sample_values: List[Any] = Field(default_factory=list)
    distinct_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    avg_value: Optional[float] = None
    null_count: Optional[int] = None
    null_percentage: Optional[float] = None
    distribution: Optional[Dict[str, int]] = None  # For categorical data with limited unique values

class TableSummary(BaseModel):
    """Summary information for a database table."""
    name: str
    row_count: int
    columns: List[ColumnSummary]
    primary_keys: List[str] = Field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = Field(default_factory=list)
    has_time_data: bool = False
    time_range: Optional[Dict[str, TimeRangeValue]] = None

class DatabaseSummary(BaseModel):
    """Summary information for the entire database."""
    database_name: str
    tables: List[TableSummary]
    total_tables: int
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    natural_language_summary: Optional[str] = None
class DatabaseSummaryAgent:
    def __init__(self, db_path: str, unique_value_threshold: int = 20):
        """
        Initialize the Database Summary Agent.
        
        Args:
            db_path: Path to the SQLite database file
            unique_value_threshold: Maximum number of unique values to consider for full enumeration
        """
        self.db_path = db_path
        self.unique_value_threshold = unique_value_threshold
        self.conn = None
        
    def connect(self) -> None:
        """Establish connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Successfully connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
            
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            
    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(tables)} tables in database")
        return tables
        
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table."""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [dict(row) for row in cursor.fetchall()]
        
        # Get foreign key information
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = [dict(row) for row in cursor.fetchall()]
        
        # Add foreign key info to columns
        for fk in foreign_keys:
            for col in columns:
                if col['name'] == fk['from']:
                    col['is_foreign_key'] = True
                    col['references'] = f"{fk['table']}.{fk['to']}"
                    
        return columns
        
    def analyze_column(self, table_name: str, column: Dict[str, Any]) -> ColumnSummary:
        """Analyze a specific column and return summary statistics."""
        if not self.conn:
            self.connect()
            
        column_name = column['name']
        cursor = self.conn.cursor()
        
        # Basic column info
        column_summary = ColumnSummary(
            name=column_name,
            data_type=column['type'],
            is_primary_key=bool(column.get('pk', 0)),
            is_foreign_key=column.get('is_foreign_key', False),
            references=column.get('references'),
            nullable=not bool(column.get('notnull', 0)),
            unique=bool(column.get('unique', 0))
        )
        
        # Count total and NULL values
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        total_rows = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL;")
        null_count = cursor.fetchone()[0]
        
        column_summary.null_count = null_count
        column_summary.null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
        
        # Get distinct count
        cursor.execute(f"SELECT COUNT(DISTINCT {column_name}) FROM {table_name};")
        column_summary.distinct_count = cursor.fetchone()[0]
        
        # Sample values
        cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 5;")
        column_summary.sample_values = [row[0] for row in cursor.fetchall()]
        
        # For numerical data
        if column['type'].lower() in ('integer', 'real', 'float', 'double', 'decimal', 'numeric'):
            cursor.execute(f"SELECT MIN({column_name}), MAX({column_name}), AVG({column_name}) FROM {table_name} WHERE {column_name} IS NOT NULL;")
            min_val, max_val, avg_val = cursor.fetchone()
            column_summary.min_value = min_val
            column_summary.max_value = max_val
            column_summary.avg_value = avg_val
            
        # For date/time data
        elif any(dt_type in column['type'].lower() for dt_type in ('date', 'time', 'datetime', 'timestamp')):
            cursor.execute(f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name} WHERE {column_name} IS NOT NULL;")
            min_date, max_date = cursor.fetchone()
            column_summary.min_value = min_date
            column_summary.max_value = max_date
            
        # For categorical data with few unique values
        if column_summary.distinct_count is not None and column_summary.distinct_count <= self.unique_value_threshold:
            cursor.execute(f"SELECT {column_name}, COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL GROUP BY {column_name};")
            distribution = {str(row[0]): row[1] for row in cursor.fetchall()}
            column_summary.distribution = distribution
            
        return column_summary
    
    def analyze_table(self, table_name: str) -> TableSummary:
        """Analyze a specific table and return summary statistics."""
        if not self.conn:
            self.connect()
            
        # Get schema information
        schema = self.get_table_schema(table_name)
        
        # Get row count
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        
        # Initialize table summary
        table_summary = TableSummary(
            name=table_name,
            row_count=row_count,
            columns=[],
            primary_keys=[],
            foreign_keys=[]
        )
        
        # Analyze each column
        has_time_data = False
        time_columns = []
        
        for column in schema:
            column_summary = self.analyze_column(table_name, column)
            table_summary.columns.append(column_summary)
            
            if column_summary.is_primary_key:
                table_summary.primary_keys.append(column_summary.name)
                
            if column_summary.is_foreign_key:
                table_summary.foreign_keys.append({
                    "column": column_summary.name,
                    "references": column_summary.references
                })
                
            # Check for time/date data
            if any(dt_type in column['type'].lower() for dt_type in ('date', 'time', 'datetime', 'timestamp')):
                has_time_data = True
                time_columns.append(column_summary.name)
                
        # Set time data flag and range if applicable
        table_summary.has_time_data = has_time_data
        if has_time_data:
            time_range = {}
            for col_name in time_columns:
                for col in table_summary.columns:
                    if col.name == col_name:
                        time_range[col_name] = {
                            "min": col.min_value,
                            "max": col.max_value
                        }
            table_summary.time_range = time_range
            
        return table_summary
    
    def analyze_database(self) -> DatabaseSummary:
        """Analyze the entire database and return comprehensive summary."""
        if not self.conn:
            self.connect()
            
        tables = self.get_tables()
        
        # Initialize database summary
        db_summary = DatabaseSummary(
            database_name=self.db_path.split('/')[-1],
            tables=[],
            total_tables=len(tables)
        )
        
        # Analyze each table
        for table_name in tables:
            table_summary = self.analyze_table(table_name)
            db_summary.tables.append(table_summary)
            
        # Find relationships between tables
        relationships = []
        for table in db_summary.tables:
            for fk in table.foreign_keys:
                relationship = {
                    "from_table": table.name,
                    "from_column": fk["column"],
                    "to_table": fk["references"].split('.')[0] if '.' in fk["references"] else "",
                    "to_column": fk["references"].split('.')[1] if '.' in fk["references"] else "",
                }
                relationships.append(relationship)
                
        db_summary.relationships = relationships
        
        return db_summary
    
    def generate_llm_summary(self, db_summary: DatabaseSummary) -> str:
        """
        Generate a natural language summary of the database using the LLM.
        
        This provides context that can be used by other agents in the pipeline.
        """
        system_prompt = """
        You are a database expert specializing in explaining database structures in clear, concise language.
        Your task is to create a natural language summary of a database based on its technical summary.
        Focus on the key characteristics that would be most relevant for data analysis:
        - Overall database structure and relationships
        - Important tables and their purposes (inferred from schema)
        - Key data points, ranges, and distributions
        - Notable patterns or characteristics
        
        Keep your summary informative but concise. Highlight anything unusual or particularly notable.
        Your report should be in the form of a markdown document.cd 
        """
        
        user_prompt = f"""
        Here is a technical summary of a database:
        
        {db_summary.model_dump_json(indent=2)}
        
        Please provide a natural language summary that explains the key characteristics of this database
        in a way that would be helpful for someone planning to analyze this data.
        """
        
        result = openai_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3
        )
        
        return result if isinstance(result, str) else "Failed to generate natural language summary."
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Main method to analyze the database and generate a complete summary.
        
        Returns:
            A dictionary containing both the structured summary and a natural language description.
        """
        try:
            # Connect to the database
            self.connect()
            
            # Analyze the database structure
            db_summary = self.analyze_database()
            
            # Generate natural language summary
            nl_summary = self.generate_llm_summary(db_summary)
            
            # Close the connection
            self.close()
            
            # Return combined results
            return {
                "technical_summary": db_summary.model_dump(),
                "natural_language_summary": nl_summary
            }
            
        except Exception as e:
            logger.error(f"Error generating database summary: {e}")
            if self.conn:
                self.close()
            raise
            
def main(db_path: str) -> Dict[str, Any]:
    """
    Entry point function to run the Database Summary Agent.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary containing the database summary
    """
    agent = DatabaseSummaryAgent(db_path)
    return agent.generate_summary()
    
if __name__ == "__main__":  
    import json
    result = main('/Users/arshath/play/experiments/insights/tests/it_sales.db')
    print(result)
    # save the result to a file 
    with open('database_summary.md', 'w') as f:
        f.write(result['natural_language_summary'])

    # save the result to a json file 
    with open('database_summary.json', 'w') as f:
        json.dump(result, f)
