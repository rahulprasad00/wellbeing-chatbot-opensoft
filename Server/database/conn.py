from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# HOST = os.getenv("DATABASE_HOST")
# DATABASE = os.getenv("DATABASE_NAME")
# USER = os.getenv("DATABASE_USER")   
# PASSWORD = os.getenv("DATABASE_PASSWORD")

# Database connection string
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
print("✅ Connected to PostgreSQL")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
