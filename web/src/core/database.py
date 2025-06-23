# Builtins
import os
from pathlib import Path
import base64
import threading
import hashlib
import sqlite3
import re
import secrets
from contextlib import contextmanager
from datetime import (
    datetime,
    timedelta,
    timezone,
    date,
    time
)
# Installed
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    validator,
    ValidationError
)
from cryptography.fernet import Fernet
# Local
from src.config import CONFIG
from src.core.logger import CORE_LOGGER
# Types
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Tuple,
    Type
)


# Pydantic Models for Database Schema
class DoctorModel(BaseModel):
    id: Optional[int] = Field(None, description="Primary key")
    name: str = Field(..., max_length=255)
    last_name: str = Field(..., max_length=255)
    email: str = Field(..., max_length=255, description="Organization email")
    password: str = Field(..., max_length=500, description="Encrypted password")
    specialization: str = Field(default="Oncología", max_length=255)

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        # Using Pydantic's built-in EmailStr for validation
        try:
            EmailStr.validate(v)
        except ValueError:
            raise ValueError('Invalid email format')
        return v
    
    class Config:
        table_name = "doctors"


class PatientModel(BaseModel):
    id: Optional[int] = Field(None, description="Primary key generated from DNI hash")
    doctor_id: int = Field(..., description="Foreign key to doctors table")
    name: str = Field(..., max_length=255)
    last_name: str = Field(..., max_length=255)
    age: int = Field(..., ge=0, le=150)
    dni: str = Field(..., pattern=r'^\d{8}[A-Za-z]$')
    is_active: bool = Field(default=True)
    creation_date: Optional[datetime] = Field(None)
    update_date: Optional[datetime] = Field(None)
    
    @validator('dni')
    def validate_dni(cls, v):
        """Validate DNI format: 8 digits + 1 letter"""
        if not re.match(r'^\d{8}[A-Za-z]$', v):
            raise ValueError('DNI must be 8 digits followed by a letter')
        return v.upper()
    
    @validator('id', pre=True, always=True)
    def generate_id_from_dni(cls, v, values):
        """Generate ID by hashing the DNI"""
        if 'dni' in values and values['dni']:
            # Create a hash from DNI and convert to positive integer
            dni_hash = hashlib.sha256(values['dni'].encode()).hexdigest()
            # Use first 8 hex chars as integer
            return int(dni_hash[:CONFIG.PATIENT_HASH_SLICE], 16)  # Take the first 8 characters of the hash and convert the hexadecimal string (base 16) into an integer
        return v
    
    class Config:
        table_name = "patients"


class PathologyModel(BaseModel):
    id: Optional[int] = Field(None, description="Auto-generated primary key")
    patient_id: int = Field(..., description="Foreign key to patients table")
    description: str = Field(..., max_length=1000)
    doctor_notes: str = Field(..., max_length=2000)
    pet_img: Optional[bytes] = Field(None, description="NIfTI image binary data")
    corrected_img: Optional[bytes] = Field(None, description="Corrected NIfTI image binary data")
    creation_date: Optional[datetime] = Field(None)
    update_date: Optional[datetime] = Field(None)
    
    class Config:
        table_name = "pathologies"


class SessionModel(BaseModel):
    id: Optional[int] = Field(None, description="Auto-generated primary key")
    doctor_id: int = Field(..., description="Foreign key to doctors table")
    session_token: str = Field(..., max_length=255, description="Unique session token")
    login_date: datetime = Field(default_factory=datetime.now)
    expires_at: datetime = Field(..., description="Session expiration time")
    is_active: bool = Field(default=True)
    
    @validator('expires_at', pre=True, always=True)
    def set_expiration(cls, v, values):
        """Set expiration to 14 hours from login_date."""
        if 'login_date' in values:
            return values['login_date'] + timedelta(hours=14)
        return datetime.now() + timedelta(hours=14)
    
    @validator('session_token', pre=True, always=True)
    def generate_token(cls, v):
        """Generate secure session token if not provided."""
        if not v:
            return secrets.token_urlsafe(32)
        return v
    
    class Config:
        table_name = "sessions"


# Class to create the tables if they dont already exist
class SchemaGenerator:
    """Generate SQL CREATE TABLE statements from Pydantic models."""
    
    TYPE_MAPPING = {
        'int': 'INTEGER',
        'float': 'REAL',
        'str': 'TEXT',
        'bool': 'BOOLEAN',
        'bytes': 'BLOB',
        'datetime': 'TIMESTAMP',
        'date': 'DATE',
        'time': 'TIME'
    }
    
    @classmethod
    def _get_sql_type(cls, field_info) -> str:
        """Convert Python type to SQLite type."""
        field_type = field_info.annotation
        
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # Get the non-None type from Optional[Type]
            non_none_types = [t for t in field_type.__args__ if t is not type(None)]
            if non_none_types:
                field_type = non_none_types[0]
        
        type_name = field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)
        return cls.TYPE_MAPPING.get(type_name, 'TEXT')
    
    @classmethod
    def _get_field_constraints(cls, field_name: str, field_info, model_class: Type[BaseModel]) -> str:
        """Generate SQL constraints for a field."""
        constraints = []
        
        # Primary key detection
        if field_name == 'id':
            if model_class == PatientModel:
                constraints.append('PRIMARY KEY')
            else:
                constraints.append('PRIMARY KEY AUTOINCREMENT')
        
        # Foreign key constraints
        if field_name == 'patient_id':
            constraints.append('REFERENCES patients(id) ON DELETE CASCADE')
        elif field_name == 'doctor_id':
            constraints.append('REFERENCES doctors(id) ON DELETE CASCADE')
        
        # Not null constraint (if field is required and not Optional)
        if field_info.is_required() and not cls._is_optional(field_info.annotation):
            constraints.append('NOT NULL')
        
        # Default values
        if field_info.default is not None and field_info.default != ...:
            if isinstance(field_info.default, str):
                constraints.append(f"DEFAULT '{field_info.default}'")
            elif isinstance(field_info.default, bool):
                constraints.append(f"DEFAULT {1 if field_info.default else 0}")
            else:
                constraints.append(f"DEFAULT {field_info.default}")
        
        # Special datetime defaults
        if field_name in ['creation_date', 'update_date']:
            constraints.append('DEFAULT CURRENT_TIMESTAMP')
        # elif field_name == 'date':
        #     constraints.append('DEFAULT CURRENT_DATE')
        # elif field_name == 'time':
        #     constraints.append('DEFAULT CURRENT_TIME')
        
        return ' '.join(constraints)
    
    @classmethod
    def _is_optional(cls, field_type) -> bool:
        """Check if a field type is Optional."""
        return (hasattr(field_type, '__origin__') and 
                field_type.__origin__ is Union and 
                type(None) in field_type.__args__)
    
    @classmethod
    def generate_create_table_sql(cls, model_class: Type[BaseModel]) -> str:
        """Generate CREATE TABLE SQL from Pydantic model."""
        table_name = getattr(model_class.Config, 'table_name', model_class.__name__.lower())
        
        field_definitions = []
        
        for field_name, field_info in model_class.__fields__.items():
            sql_type = cls._get_sql_type(field_info)
            constraints = cls._get_field_constraints(field_name, field_info, model_class)
            
            field_def = f"{field_name} {sql_type}"
            if constraints:
                field_def += f" {constraints}"
            
            field_definitions.append(field_def)
        
        # Add indexes
        indexes = []
        if model_class == PatientModel:
            indexes.extend([
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_dni ON {table_name}(dni)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}(name, last_name)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_doctor_id ON {table_name}(doctor_id)"
            ])
        elif model_class == PathologyModel:
            indexes.extend([
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_patient_id ON {table_name}(patient_id)",
                # f"CREATE INDEX IF NOT EXISTS idx_{table_name}_doctor_id ON {table_name}(doctor_id)"
                # f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)"
            ])
        elif model_class == DoctorModel:
            indexes.append(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}(name, last_name)")
        
        fields_str = ',\n            '.join(field_definitions)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {fields_str}
        )
        """
        
        return create_table_sql.strip(), indexes


# Singleton class to manage the database setup and operations
class DatabaseManager:
    """
        Singleton SQLite database manager that handles connection pooling,
        table initialization, and query execution.
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    MODELS = [DoctorModel, PatientModel, PathologyModel, SessionModel]
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.db_path = CONFIG.DB_PATH
                    self.db_key_path = CONFIG.DB_KEY_PATH
                    self.logger = CORE_LOGGER
                    self._connection_pool = threading.local()
                    self._ensure_database_exists()
                    self._initialize_tables_from_models()
                    DatabaseManager._initialized = True
    
    def _ensure_database_exists(self) -> None:
        """Ensure the database file and directory exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the database file if it doesn't exist
        if not self.db_path.exists():
            self.logger.info(f"Creating new database at: {self.db_path}")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")  # Simple query to create the file
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._connection_pool, 'connection'):
            self._connection_pool.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable foreign keys and set pragmas for better performance
            self._connection_pool.connection.execute("PRAGMA foreign_keys = ON")
            self._connection_pool.connection.execute("PRAGMA journal_mode = WAL")
            self._connection_pool.connection.execute("PRAGMA synchronous = NORMAL")
            self._connection_pool.connection.row_factory = sqlite3.Row
        
        return self._connection_pool.connection
    
    # Add these methods to the DatabaseManager class
    def _get_encryption_key(self) -> bytes:
        """Get or create encryption key for password encryption."""
        if self.db_key_path.exists():
            with open(self.db_key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(self.db_key_path, 'wb') as f:
                f.write(key)
            # Make key file read-only
            os.chmod(self.db_key_path, 0o600)
            return key

    def _initialize_tables_from_models(self):
        """Initialize database tables from Pydantic models."""
        try:
            with self.get_connection() as conn:
                for model_class in self.MODELS:
                    table_name = getattr(model_class.Config, 'table_name', model_class.__name__.lower())
                    
                    # Check if table already exists
                    if not self.table_exists(table_name):
                        self.logger.info(f"Creating table: {table_name}")
                        
                        # Generate and execute CREATE TABLE statement
                        create_sql, indexes = SchemaGenerator.generate_create_table_sql(model_class)
                        conn.execute(create_sql)

                        # Create indexes
                        for index_sql in indexes:
                            conn.execute(index_sql)
                        
                        self.logger.info(f"Table {table_name} created successfully")
                    else:
                        self.logger.info(f"Table {table_name} already exists, skipping creation")
                
                conn.commit()
                self.logger.info("Database schema initialization completed")
            
            self._initialize_sample_data()
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing tables: {e}")
            raise
    
    def _initialize_sample_data(self):
        """Insert sample data if tables are empty."""
        try:
            with self.get_connection() as conn:
                # Check if doctors table is empty and add sample doctors
                doctor_count = self.execute_query("SELECT COUNT(*) as count FROM doctors", fetch="one")
                if doctor_count['count'] == 0:
                    sample_doctors = [
                        ("Angel", "Torrado", "atorrado@fakehospital.com", "1234asdf"),
                        ("Javier", "Hernandez", "jhernandez@fakehospital.com", "asdf1234"),
                    ]
                    doctor_ids = []
                    for name, last_name, email, password in sample_doctors:
                        encrypted_password = self.encrypt_password(password)
                        doctor_id = self.execute_query(
                            "INSERT INTO doctors (name, last_name, email, password) VALUES (?, ?, ?, ?)",
                            (name, last_name, email, encrypted_password)
                        )
                        doctor_ids.append(doctor_id)
                    self.logger.info(f"Inserted {len(sample_doctors)} sample doctors")

                # Check if patients table is empty and add sample patients
                patient_count = self.execute_query("SELECT COUNT(*) as count FROM patients", fetch="one")
                if patient_count['count'] == 0:
                    sample_patients = [
                        ("Carlos", "Llueca", 25, "51502310C", doctor_ids[1]),
                        ("Carlos", "Martínez", 30, "12345678A", doctor_ids[1]),
                        ("Laura", "Gómez", 28, "23456789B", doctor_ids[1]),
                        ("Javier", "Pérez", 35, "34567890C", doctor_ids[1]),
                        ("Marta", "Sánchez", 22, "45678901D", doctor_ids[0]),
                        ("Sofía", "López", 27, "56789012E", doctor_ids[0]),
                        ("David", "Fernández", 40, "67890123F", doctor_ids[0]),
                        ("Ana", "Ramírez", 31, "78901234G", doctor_ids[0]),
                        ("Luis", "Torres", 29, "89012345H", doctor_ids[0]),
                        ("Elena", "Hernández", 24, "90123456I", doctor_ids[0]),
                        ("Pedro", "Torres", 33, "01234567J", doctor_ids[0])
                    ]
                    
                    for name, last_name, age, dni, doc_id in sample_patients:
                        # Generate ID from DNI like the PatientModel does
                        dni_hash = hashlib.sha256(dni.encode()).hexdigest()
                        patient_id = int(dni_hash[:CONFIG.PATIENT_HASH_SLICE], 16)
                        
                        self.execute_query(
                            """INSERT INTO patients (id, doctor_id, name, last_name, age, dni, creation_date, update_date) 
                            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                            (patient_id, doc_id, name, last_name, age, dni)
                        )
                    self.logger.info(f"Inserted {len(sample_patients)} sample patients")
                    
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting sample data: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database transaction rolled back: {e}")
            raise
        finally:
            # Don't close the connection as it's thread-local and reused
            pass
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Union[Tuple, Dict[str, Any]]] = None,
        fetch: str = "none"
    ) -> Optional[Union[List[sqlite3.Row], sqlite3.Row, int]]:
        """
        Execute a SQL query with optional parameters.
        
        Args:
            query: SQL query string
            params: Query parameters (tuple for positional, dict for named)
            fetch: 'none', 'one', 'all', or 'rowcount'
        
        Returns:
            Query results based on fetch parameter
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                if fetch == "one":
                    return cursor.fetchone()
                elif fetch == "all":
                    return cursor.fetchall()
                elif fetch == "rowcount":
                    return cursor.rowcount
                elif fetch == "rowcount-update":
                    conn.commit()
                    return cursor.rowcount
                else:
                    conn.commit()
                    return cursor.lastrowid if query.strip().upper().startswith('INSERT') else None
                    
        except sqlite3.Error as e:
            self.logger.error(f"Database query error: {e}")
            self.logger.error(f"Query: {query}")
            self.logger.error(f"Params: {params}")
            raise
    
    def execute_many(self, query: str, params_list: List[Union[Tuple, Dict[str, Any]]]) -> int:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter sets
        
        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
        except sqlite3.Error as e:
            self.logger.error(f"Database executemany error: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> List[sqlite3.Row]:
        """Get table schema information."""
        return self.execute_query(f"PRAGMA table_info({table_name})", fetch="all")
    
    def get_all_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        result = self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            fetch="all"
        )
        return [row['name'] for row in result] if result else []
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
            fetch="one"
        )
        return result is not None
    
    def get_row_count(self, table_name: str, where_clause: str = "", params: Optional[Tuple] = None) -> int:
        """Get row count for a table with optional WHERE clause."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        result = self.execute_query(query, params, fetch="one")
        return result['count'] if result else 0
    
    def close_all_connections(self):
        """Close all thread-local connections. Use with caution."""
        if hasattr(self._connection_pool, 'connection'):
            try:
                self._connection_pool.connection.close()
                delattr(self._connection_pool, 'connection')
                self.logger.info("Database connections closed")
            except Exception as e:
                self.logger.error(f"Error closing database connections: {e}")
    
    def vacuum(self):
        """Run VACUUM to optimize database file."""
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            self.logger.info("Database VACUUM completed")
        except sqlite3.Error as e:
            self.logger.error(f"Error running VACUUM: {e}")
            raise
    
    def backup(self, backup_path: Union[str, Path]):
        """Create a backup of the database."""
        backup_path = Path(backup_path)
        try:
            with self.get_connection() as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
            self.logger.info(f"Database backup created at: {backup_path}")
        except sqlite3.Error as e:
            self.logger.error(f"Error creating backup: {e}")
            raise

    def encrypt_password(self, password: str) -> str:
        """Encrypt a password string."""
        key = self._get_encryption_key()
        f = Fernet(key)
        encrypted_password = f.encrypt(password.encode())
        return base64.b64encode(encrypted_password).decode()

    def decrypt_password(self, encrypted_password: str) -> str:
        """Decrypt an encrypted password string."""
        key = self._get_encryption_key()
        f = Fernet(key)
        encrypted_bytes = base64.b64decode(encrypted_password.encode())
        decrypted_password = f.decrypt(encrypted_bytes)
        return decrypted_password.decode()


# Convenience function to get the singleton instance
def get_db_manager() -> DatabaseManager:
    """Get the singleton DatabaseManager instance."""
    return DatabaseManager()


# Class that contains helper functions for CRUD operations
class DatabaseOperations:
    """Helper class with CRUD operations for your models."""
    
    def __init__(self):
        self.db = get_db_manager()
    
    # Doctor operations
    def create_doctor(self, name: str, last_name: str, email: str, password: str, specialization: str="Oncología") -> int:
        """Create a new doctor."""
        doctor_data = {
            'name': name,
            'last_name': last_name,
            'email': email,
            'password': self.db.encrypt_password(password),
            'specialization': specialization
        } 
        try:
            new_doctor = DoctorModel(**doctor_data)
        except ValidationError as e:
            self.db.logger.error(f"Error inserting doctor: {e}")
            return None
        return self.db.execute_query(
            "INSERT INTO doctors (name, last_name, email, password, specialization) VALUES (?, ?, ?, ?, ?)",
            (new_doctor.name, new_doctor.last_name, new_doctor.email, new_doctor.password, new_doctor.specialization)
        )
    
    def get_doctor_by_id(self, doctor_id: int) -> Optional[sqlite3.Row]:
        """Get doctor by ID."""
        return self.db.execute_query(
            "SELECT * FROM doctors WHERE id = ?",
            (doctor_id,),
            fetch="one"
        )
    
    def get_doctor_by_email(self, email: str) -> Optional[sqlite3.Row]:
        """Get doctor by email."""
        return self.db.execute_query(
            "SELECT id, password FROM doctors WHERE email = ?",
            (email,),
            fetch="one"
        )
    
    def verify_password(self, doctor_id: int, password: str) -> bool:
        """Verify a doctor's password."""
        doctor = self.get_doctor_by_id(doctor_id)
        if not doctor:
            return False
        
        try:
            decrypted_password = self.db.decrypt_password(doctor['password'])
        except Exception:
            return False

        return decrypted_password == password

    def update_password(self, doctor_id: int, new_password: str) -> bool:
        """Update a docotr's password"""
        doctor = self.get_doctor_by_id(doctor_id)
        if not doctor:
            return False
        
        try:
            encrypted_password = self.db.encrypt_password(new_password)
            values = [encrypted_password, doctor_id]
            affected_rows = self.db.execute_query(
                f"UPDATE doctors SET password = ? WHERE id = ?",
                tuple(values),
                fetch='rowcount'
            )
        except Exception:
            return False

        return True if affected_rows == 1 else False

    # Patient operations
    def create_patient(self, name: str, last_name: str, age: int, dni: str, is_active: bool = True) -> int:
        """Create a new patient."""
        # Validate and create patient model to get the generated ID
        patient_data = {
            'name': name,
            'last_name': last_name,
            'age': age,
            'dni': dni,
            'is_active': is_active
        }
        try:
            patient = PatientModel(**patient_data)
        except ValidationError as e:
            self.db.logger.error(f"Error inserting patient: {e}")
            return None
        return self.db.execute_query(
            """INSERT INTO patients (id, name, last_name, age, dni, is_active, creation_date, update_date) 
               VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            (patient.id, patient.name, patient.last_name, patient.age, patient.dni, patient.is_active)
        )
    
    def get_patient_by_dni(self, dni: str) -> Optional[sqlite3.Row]:
        """Get patient by DNI."""
        return self.db.execute_query(
            "SELECT id, name, last_name, age FROM patients WHERE dni = ?",
            (dni.upper(),),
            fetch="one"
        )
    
    def get_patient_by_id(self, patient_id: str) -> Optional[sqlite3.Row]:
        """Get patient by ID."""
        return self.db.execute_query(
            "SELECT name, last_name, age, dni FROM patients WHERE id = ? and is_active = ?",
            (patient_id, True),
            fetch="one"
        )
    
    def update_patient(self, patient_id: int, **kwargs) -> bool:
        """Update patient fields."""
        if not kwargs:
            return False
        
        # Always update the update_date
        kwargs['update_date'] = datetime.now()
        
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        values = list(kwargs.values()) + [patient_id]
        
        affected_rows = self.db.execute_query(
            f"UPDATE patients SET {set_clause} WHERE id = ?",
            tuple(values),
            fetch="rowcount"
        )
        return affected_rows > 0
    
    def get_all_patients_by_doctor(self, doctor_id: int) -> Optional[sqlite3.Row]:
        """Gets all the patients from the database"""
        return self.db.execute_query(
            "SELECT id, name, last_name FROM patients WHERE doctor_id = ?",
            (doctor_id,),
            fetch="all"
        )

    # Pathology operations
    def create_pathology(
        self,
        patient_id: int,
        description: str, 
        doctor_notes: str,
        pet_img: bytes = None,
        corrected_img: bytes = None
    ) -> int:
        """Create a new pathology record."""
        # Validate and create pathology model to ensure data integrity
        pathology_data = {
            'patient_id': patient_id,
            'description': description,
            'doctor_notes': doctor_notes,
            'pet_img': pet_img,
            'corrected_img': corrected_img
        }
        
        try:
            pathology = PathologyModel(**pathology_data)
        except ValidationError as e:
            self.db.logger.error(f"Error inserting pathology: {e}")
            return None
        return self.db.execute_query(
            """INSERT INTO pathologies (patient_id, description, doctor_notes, 
            pet_img, corrected_img, creation_date, update_date) 
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            (pathology.patient_id, pathology.description, pathology.doctor_notes, pathology.pet_img, pathology.corrected_img)
        )
    
    def get_pathologies_by_patient(self, patient_id: int) -> List[sqlite3.Row]:
        """Get all pathologies for a patient."""
        return self.db.execute_query(
            """SELECT id, description, doctor_notes, pet_img, corrected_img
               FROM pathologies
               WHERE patient_id = ? 
               ORDER BY creation_date DESC""",
            (patient_id,),
            fetch="all"
        )
    
    def get_pathology_by_id(self, pathology_id: int, patient_id: int) -> List[sqlite3.Row]:
        """Get all pathologies for a patient."""
        return self.db.execute_query(
            """SELECT id, description, doctor_notes, pet_img, corrected_img
               FROM pathologies
               WHERE id = ? 
               AND patient_id = ?""",
            (pathology_id, patient_id),
            fetch="one"
        )
    
    def update_original_pet_image(self, pathology_id: int, image_bytes: bytes) -> bool:
        """Update pathology record with image data."""
        try:
            result = self.db.execute_query(
                "UPDATE pathologies SET pet_img = ?, update_date = CURRENT_TIMESTAMP WHERE id = ?",
                (image_bytes, pathology_id),
                fetch="rowcount-update"
            )
            return result > 0
        except Exception as e:
            self.db.logger.error(f"Error updating original PET image: {e}")
            return False
        
    def update_corrected_pet_image(self, pathology_id: int, image_bytes: bytes) -> bool:
        """Update pathology record with image data."""
        try:
            result = self.db.execute_query(
                "UPDATE pathologies SET corrected_img = ?, update_date = CURRENT_TIMESTAMP WHERE id = ?",
                (image_bytes, pathology_id),
                fetch="rowcount-update"
            )
            return result > 0
        except Exception as e:
            self.db.logger.error(f"Error updating corrected PET image: {e}")
            return False

    # Session operations
    def create_session(self, doctor_id: int) -> str:
        """Create a new session for a doctor and return the session token."""
        # First, check if the doctor has an active session and use that
        active_session = self.get_session_by_id(doctor_id)
        if active_session:
            return active_session['session_token']
        
        # If no active sessions, invalidate any existing expired sessions for this doctor
        self.db.execute_query(
            "UPDATE sessions SET is_active = ? WHERE doctor_id = ? AND is_active = ?",
            (False, doctor_id, True)
        )

        self.cleanup_expired_sessions()
        
        session_token = secrets.token_urlsafe(32)
        login_date = datetime.now()
        expires_at = login_date + timedelta(hours=14)
        
        self.db.execute_query(
            """INSERT INTO sessions (doctor_id, session_token, login_date, expires_at, is_active) 
            VALUES (?, ?, ?, ?, ?)""",
            (doctor_id, session_token, login_date, expires_at, True)
        )
        return session_token

    def get_session_by_token(self, session_token: str) -> Optional[sqlite3.Row]:
        """Get active session by token."""
        return self.db.execute_query(
            """SELECT s.session_token, d.id, d.name, d.last_name, d.email
            FROM sessions s 
            JOIN doctors d ON s.doctor_id = d.id 
            WHERE s.session_token = ? AND s.is_active = ? AND s.expires_at > ?""",
            (session_token, True, datetime.now()),
            fetch="one"
        )
    
    def get_session_by_id(self, doctor_id: int) -> Optional[sqlite3.Row]:
        """Get the current active session for a doctor."""
        return self.db.execute_query(
            "SELECT session_token FROM sessions WHERE doctor_id = ? AND is_active = ? AND expires_at > ?",
            (doctor_id, True, datetime.now()),
            fetch="one"
        )

    def get_doctor_session_history(self, doctor_id: int) -> List[sqlite3.Row]:
        """Get all sessions (active and inactive) for a doctor."""
        return self.db.execute_query(
            "SELECT * FROM sessions WHERE doctor_id = ? ORDER BY login_date DESC",
            (doctor_id,),
            fetch="all"
        )

    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session (logout)."""
        affected_rows = self.db.execute_query(
            "UPDATE sessions SET is_active = ? WHERE session_token = ?",
            (False, session_token),
            fetch="rowcount"
        )
        self.cleanup_expired_sessions()
        return affected_rows > 0

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from database."""
        affected_rows = self.db.execute_query(
            "DELETE FROM sessions WHERE expires_at < ? OR is_active = ?",
            (datetime.now(), False),
            fetch="rowcount"
        )
        return affected_rows

    def get_doctor_sessions(self, doctor_id: int) -> List[sqlite3.Row]:
        """Get all active sessions for a doctor."""
        return self.db.execute_query(
            "SELECT * FROM sessions WHERE doctor_id = ? AND is_active = ? AND expires_at > ?",
            (doctor_id, True, datetime.now()),
            fetch="all"
        )

    def extend_session(self, session_token: str) -> bool:
        """Extend session by another 14 hours."""
        new_expiry = datetime.now() + timedelta(hours=14)
        affected_rows = self.db.execute_query(
            "UPDATE sessions SET expires_at = ? WHERE session_token = ? AND is_active = ?",
            (new_expiry, session_token, True),
            fetch="rowcount"
        )
        return affected_rows > 0

