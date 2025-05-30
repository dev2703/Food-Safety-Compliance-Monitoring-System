import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'port': os.getenv('DB_PORT', '5432'),
    'dbname': os.getenv('DB_NAME', 'sentinel')
}

def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_CONFIG['dbname'],))
        exists = cur.fetchone()
        
        if not exists:
            logger.info(f"Creating database {DB_CONFIG['dbname']}...")
            cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
            logger.info("Database created successfully")
        else:
            logger.info(f"Database {DB_CONFIG['dbname']} already exists")
            
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise

def create_tables():
    """Create the necessary tables in the database."""
    try:
        # Connect to the database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Read and execute schema.sql
        schema_path = Path(__file__).parent.parent / 'schema.md'
        with open(schema_path, 'r') as f:
            schema_content = f.read()
            
        # Extract SQL statements from schema.md
        sql_statements = []
        current_statement = []
        in_sql_block = False
        
        for line in schema_content.split('\n'):
            if line.startswith('```sql'):
                in_sql_block = True
                continue
            elif line.startswith('```') and in_sql_block:
                in_sql_block = False
                if current_statement:
                    sql_statements.append('\n'.join(current_statement))
                    current_statement = []
                continue
                
            if in_sql_block:
                current_statement.append(line)
        
        # Execute each SQL statement
        for statement in sql_statements:
            try:
                cur.execute(statement)
                logger.info("Successfully executed SQL statement")
            except Exception as e:
                logger.error(f"Error executing SQL statement: {str(e)}")
                logger.error(f"Statement: {statement}")
                raise
        
        conn.commit()
        logger.info("All tables created successfully")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        raise

def main():
    """Main function to set up the database and create tables."""
    try:
        logger.info("Starting database setup...")
        create_database()
        create_tables()
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 