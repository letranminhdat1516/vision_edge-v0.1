import os
import sys
from sqlalchemy import create_engine, MetaData, inspect
from dotenv import load_dotenv

load_dotenv()

class DatabaseMigration:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.engine = create_engine(self.db_url)
        self.inspector = inspect(self.engine)
    
    def get_table_info(self, table_name):
        columns = self.inspector.get_columns(table_name)
        return columns
    
    def map_postgres_to_sqlalchemy(self, pg_type):
        pg_str = str(pg_type).lower()
        
        if 'uuid' in pg_str:
            return 'UUID(as_uuid=True)'
        elif 'varchar' in pg_str or 'character varying' in pg_str:
            return 'String'
        elif 'text' in pg_str:
            return 'Text'
        elif 'timestamp' in pg_str:
            return 'DateTime'
        elif 'boolean' in pg_str:
            return 'Boolean'
        elif 'integer' in pg_str or 'bigint' in pg_str:
            return 'Integer'
        elif 'numeric' in pg_str or 'decimal' in pg_str:
            return 'DECIMAL'
        elif 'json' in pg_str:
            return 'JSON'
        else:
            return 'String'
    
    def generate_model_code(self):
        tables = self.inspector.get_table_names()
        
        imports = """from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey, JSON, DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

"""
        
        models_code = imports
        
        for table_name in tables:
            columns = self.get_table_info(table_name)
            class_name = ''.join(word.capitalize() for word in table_name.split('_'))
            
            models_code += f"class {class_name}(Base):\n"
            models_code += f"    __tablename__ = '{table_name}'\n\n"
            
            for col in columns:
                col_type = self.map_postgres_to_sqlalchemy(col['type'])
                
                line = f"    {col['name']} = Column({col_type}"
                
                if col.get('primary_key'):
                    line += ", primary_key=True"
                
                if not col.get('nullable', True):
                    line += ", nullable=False"
                
                if col.get('default') and 'gen_random_uuid' in str(col['default']):
                    line += ", default=uuid.uuid4"
                
                line += ")\n"
                models_code += line
            
            models_code += "\n"
        
        return models_code
    
    def create_models_file(self):
        os.makedirs('models', exist_ok=True)
        models_content = self.generate_model_code()
        
        with open('models/database_models.py', 'w', encoding='utf-8') as f:
            f.write(models_content)
    
    def create_orm_file(self):
        orm_content = """from sqlalchemy import create_engine
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
"""
        
        with open('models/simple_orm.py', 'w', encoding='utf-8') as f:
            f.write(orm_content)
    
    def create_init_file(self):
        init_content = """from .database_models import *
from .simple_orm import orm
"""
        
        with open('models/__init__.py', 'w', encoding='utf-8') as f:
            f.write(init_content)
    
    def run_migration(self):
        self.create_models_file()
        self.create_orm_file()
        self.create_init_file()

if __name__ == "__main__":
    migration = DatabaseMigration()
    migration.run_migration()