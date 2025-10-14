#!/usr/bin/env python3
"""
SQLAlchemy Model Generator
Generates SQLAlchemy models from existing database schema
"""

import os
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_models():
    """Generate SQLAlchemy models from existing database"""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found in .env file")
        return
    
    print(f"🔗 Connecting to database...")
    
    try:
        # Create engine and reflect existing database
        engine = create_engine(database_url)
        metadata = MetaData()
        
        print("📊 Reflecting database schema...")
        metadata.reflect(bind=engine)
        
        print(f"✅ Found {len(metadata.tables)} tables:")
        for table_name in metadata.tables.keys():
            print(f"   📋 {table_name}")
        
        # Generate models directory
        models_dir = "src/models/generated"
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate base models file
        generate_base_models(metadata, models_dir)
        
        # Generate individual model files
        generate_individual_models(metadata, models_dir)
        
        print(f"✅ Models generated in {models_dir}/")
        
    except Exception as e:
        print(f"❌ Error generating models: {e}")

def generate_base_models(metadata, output_dir):
    """Generate base models file with all tables"""
    
    models_content = '''"""
Generated SQLAlchemy Models
Auto-generated from database schema
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

'''
    
    # Generate each table as a model
    for table_name, table in metadata.tables.items():
        class_name = table_name_to_class_name(table_name)
        
        models_content += f'''
class {class_name}(Base):
    __tablename__ = '{table_name}'
    
'''
        
        # Generate columns
        for column in table.columns:
            column_def = generate_column_definition(column)
            models_content += f"    {column_def}\n"
        
        models_content += "\n"
    
    # Write to file
    with open(f"{output_dir}/models.py", "w", encoding="utf-8") as f:
        f.write(models_content)
    
    print(f"📝 Generated base models file: {output_dir}/models.py")

def generate_individual_models(metadata, output_dir):
    """Generate individual model files for each table"""
    
    for table_name, table in metadata.tables.items():
        class_name = table_name_to_class_name(table_name)
        
        model_content = f'''"""
{class_name} Model
Generated from table: {table_name}
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class {class_name}(Base):
    __tablename__ = '{table_name}'
    
'''
        
        # Generate columns
        for column in table.columns:
            column_def = generate_column_definition(column)
            model_content += f"    {column_def}\n"
        
        model_content += f'''
    def __repr__(self):
        return f"<{class_name}(id={{self.id}})>"
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {{c.name: getattr(self, c.name) for c in self.__table__.columns}}
'''
        
        # Write individual model file
        filename = f"{output_dir}/{table_name}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(model_content)
        
        print(f"📄 Generated model: {filename}")

def generate_column_definition(column):
    """Generate SQLAlchemy column definition from database column"""
    
    column_name = column.name
    column_type = str(column.type)
    
    # Map database types to SQLAlchemy types
    type_mapping = {
        'INTEGER': 'Integer',
        'VARCHAR': 'String',
        'TEXT': 'Text',
        'BOOLEAN': 'Boolean',
        'TIMESTAMP': 'DateTime',
        'FLOAT': 'Float',
        'UUID': 'UUID(as_uuid=True)',
    }
    
    # Determine SQLAlchemy type
    sa_type = 'String'  # default
    for db_type, sa_type_mapped in type_mapping.items():
        if db_type in column_type.upper():
            sa_type = sa_type_mapped
            break
    
    # Handle string length
    if 'VARCHAR' in column_type.upper() and '(' in column_type:
        length = column_type.split('(')[1].split(')')[0]
        sa_type = f"String({length})"
    
    # Build column definition
    parts = [f"{column_name} = Column({sa_type}"]
    
    # Add constraints
    if column.primary_key:
        parts.append("primary_key=True")
    
    if not column.nullable:
        parts.append("nullable=False")
    
    if column.default is not None:
        if hasattr(column.default, 'arg'):
            default_val = repr(column.default.arg)
            parts.append(f"default={default_val}")
    
    return ", ".join(parts) + ")"

def table_name_to_class_name(table_name):
    """Convert table name to PascalCase class name"""
    # Split by underscore and capitalize each part
    parts = table_name.split('_')
    return ''.join(word.capitalize() for word in parts)

if __name__ == "__main__":
    print("🏗️ SQLAlchemy Model Generator")
    print("=" * 50)
    generate_models()
    print("=" * 50)
    print("✅ Model generation completed!")