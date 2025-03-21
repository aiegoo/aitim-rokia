from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import pandas as pd
from uuid import UUID

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Database:
    def __init__(self):
        self.engine = engine

    def get_admin_uuid():
        """Fetch the UUID of the admin user from PostgreSQL."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT id FROM users WHERE username = :username"), {"username": "admin"})
                admin_data = result.fetchone()
                if admin_data:
                    return str(admin_data[0])  # Convert UUID to string
                else:
                    raise ValueError("Admin user not found in database.")
        except Exception as e:
            raise ValueError(f"Error fetching admin UUID: {str(e)}")

    def execute_query(self, query, params=None):
        try:
            # Convert 'user_id' to UUID if it exists
            if params and "user_id" in params:
                params["user_id"] = UUID(params["user_id"])  # Ensure it's a valid UUID
            
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params) if params else connection.execute(text(query))
                connection.commit()
                return result
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise

    def fetch_all(self, query, params=None):
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query), params) if params else connection.execute(text(query))
                return result.fetchall()
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise

    def fetch_as_dataframe(self, query, params=None):
        try:
            return pd.read_sql_query(text(query), self.engine, params=params)
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise

