from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .database_models import Base
import os
from dotenv import load_dotenv

load_dotenv()

class SimpleORM:
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.Session()
    
    def save(self, obj):
        session = self.get_session()
        try:
            session.add(obj)
            session.commit()
            return obj
        finally:
            session.close()
    
    def query(self, model_class):
        session = self.get_session()
        return session.query(model_class)
    
    def execute_raw(self, sql, params=None):
        with self.engine.connect() as conn:
            return conn.execute(sql, params or {})

orm = SimpleORM()
