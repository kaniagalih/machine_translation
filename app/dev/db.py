import psycopg2
import os
from psycopg2 import pool
from dotenv import load_dotenv

# Database connection parameters
# Replace these with your actual PostgreSQL connection details

load_dotenv()

DB_PARAMS = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

# Create a connection pool for efficient database connections
connection_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **DB_PARAMS)

def create_table():
    """
    Create the translations table in PostgreSQL if it doesn't exist.
    """
    try:
        # Get a connection from the pool
        conn = connection_pool.getconn()
        
        # Create a cursor object
        with conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    jawa_text TEXT NOT NULL,
                    indonesia_text TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    expected TEXT NOT NULL,
                    suggestion TEXT NOT NULL
                    
                )
            """)
        
        # Commit the transaction
        conn.commit()
    
    except (Exception, psycopg2.Error) as error:
        print(f"Error creating table: {error}")
    
    finally:
        # Return the connection to the pool
        if conn:
            connection_pool.putconn(conn)

def insert_data(name, age, jawa_text, indonesia_text, rating, expected, suggestion):
    """
    Insert a new translation record into the database.
    
    :param name: Name of the translator
    :param age: Age of the translator
    :param jawa_text: Text in Javanese
    :param indonesia_text: Text in Indonesian
    :param rating: Rating of the translation
    :param expected: Correct translation from user
    :param suggestion: Translator's suggestions
    """
    try:
        # Get a connection from the pool
        conn = connection_pool.getconn()
        
        # Create a cursor object
        with conn.cursor() as c:
            c.execute(
                """
                INSERT INTO translations 
                (name, age, jawa_text, indonesia_text, rating, expected, suggestion) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, 
                (name, age, jawa_text, indonesia_text, rating, expected, suggestion)
            )
        
        # Commit the transaction
        conn.commit()
    
    except (Exception, psycopg2.Error) as error:
        print(f"Error inserting data: {error}")
    
    finally:
        # Return the connection to the pool
        if conn:
            connection_pool.putconn(conn)

def close_connection_pool():
    """
    Closes all database connections in the pool.
    Call this when your application is shutting down.
    """
    if connection_pool:
        connection_pool.closeall()