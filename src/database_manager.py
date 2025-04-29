import os
import sqlite3
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    DatabaseManager is a class responsible for managing interactions with an SQLite database.
    It provides methods to initialize the database, store and retrieve data related to user interactions,
    responses, and analysis, as well as manage user accounts.
    
    Attributes:
        db_path (str): The file path to the SQLite database.
    """
    def __init__(self, db_path = os.path.join(os.path.dirname(__file__), "../agent_history.db")):
        """
        Initializes the DatabaseManager instance.

        Args:
            db_path (str): The file path to the database. Defaults to a path
                pointing to "../agent_history.db" relative to the current file.

        Attributes:
            db_path (str): The file path to the database.

        Calls:
            initialize_db(): Sets up the database if it does not already exist.
        """
        self.db_path = db_path
        self.initialize_db()
    
    def initialize_db(self):
        """
        Initializes the database by creating necessary tables if they do not already exist.
        Tables created:
        - interactions: Stores interaction details such as job ID, query, domain, question type, timestamp, username, and roles.
        - responses: Stores agent responses linked to interactions, including job ID, agent ID, response text, aggregator status, and timestamp.
        - analysis: Stores analysis data for interactions, including consensus score, analysis data, and timestamp.
        - users: Stores user credentials with unique usernames and passwords.
        Logs a success message upon successful initialization or an error message if an exception occurs.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create interactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                query TEXT NOT NULL,
                domain TEXT,
                question_type TEXT,
                timestamp INTEGER NOT NULL,
                username TEXT NOT NULL,
                roles TEXT
            )
            ''')
            
            # Create responses table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                response TEXT NOT NULL,
                is_aggregator BOOLEAN NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (job_id) REFERENCES interactions(job_id)
            )
            ''')
            
            # Create analysis table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis (
                job_id TEXT PRIMARY KEY,
                consensus_score REAL,
                analysis_data TEXT,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (job_id) REFERENCES interactions(job_id)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def get_user_history(self, username: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the interaction history of a specific user from the database.
        
        Args:
            username (str): The username of the user whose history is to be retrieved.
            limit (int, optional): The maximum number of interactions to retrieve. Defaults to 50.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents an interaction.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
            SELECT i.job_id, i.query, i.domain, i.question_type, i.timestamp, i.roles, a.consensus_score
            FROM interactions i
            LEFT JOIN analysis a ON i.job_id = a.job_id
            WHERE i.username = ?
            ORDER BY i.timestamp DESC
            LIMIT ?
            """, (username, limit))

            interactions = []
            for row in cursor.fetchall():
                interaction = dict(row)

                resp_cursor = conn.cursor()
                resp_cursor.execute(
                    "SELECT agent_id, response, is_aggregator FROM responses WHERE job_id = ?",
                    (interaction['job_id'],)
                )
                responses = {r['agent_id']: r['response'] for r in resp_cursor.fetchall()}
                interaction['responses'] = responses
                interactions.append(interaction)

            conn.close()
            return interactions
        except Exception as e:
            logger.error(f"Error getting user history: {str(e)}")
            return []

    def save_interaction(self, job_id: str, query: str, domain: str, question_type: str, username: str, roles: Optional[str] = "") -> bool:
        """
        Saves an interaction record to the database.
        
        Args:
            job_id (str): The unique identifier for the job associated with the interaction.
            query (str): The query or input provided by the user.
            domain (str): The domain or category of the interaction.
            question_type (str): The type of question being asked.
            username (str): The username of the individual initiating the interaction.
            roles (Optional[str]): Additional roles or metadata associated with the interaction. Defaults to an empty string.
        
        Returns:
            bool: True if the interaction was successfully saved, False otherwise.
        
        Raises:
            Logs an error message if an exception occurs during the database operation.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = int(time.time())
            
            cursor.execute(
                "INSERT INTO interactions (job_id, query, domain, question_type, timestamp, username, roles) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, query, domain, question_type, timestamp, username, roles)
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving interaction: {str(e)}")
            return False
    
    def save_responses(self, job_id: str, responses: Dict[str, str], aggregator_id: Optional[str] = None) -> bool:
        """
        Saves agent responses to the database for a specific job.
        
        Args:
            job_id (str): The unique identifier for the job.
            responses (Dict[str, str]): A dictionary mapping agent IDs to their responses.
            aggregator_id (Optional[str]): The ID of the aggregator agent, if any. Defaults to None.
        
        Returns:
            bool: True if the responses were successfully saved, False otherwise.
        
        Raises:
            Exception: Logs an error message if an exception occurs during the database operation.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = int(time.time())
            
            for agent_id, response in responses.items():
                is_aggregator = agent_id == aggregator_id
                cursor.execute(
                    "INSERT INTO responses (job_id, agent_id, response, is_aggregator, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (job_id, agent_id, response, is_aggregator, timestamp)
                )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving responses: {str(e)}")
            return False
    
    def save_analysis(self, job_id: str, consensus_score: float, analysis_data: Dict[str, Any]) -> bool:
        """
        Saves the analysis data for a specific job into the database.
        
        Args:
            job_id (str): The unique identifier for the job.
            consensus_score (float): The consensus score associated with the analysis.
            analysis_data (Dict[str, Any]): A dictionary containing the analysis data to be saved.
        
        Returns:
            bool: True if the analysis data was successfully saved, False otherwise.
        
        Raises:
            Exception: Logs any exception that occurs during the database operation.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = int(time.time())
            
            # Convert analysis data to JSON
            analysis_json = json.dumps(analysis_data)
            
            cursor.execute(
                "INSERT INTO analysis (job_id, consensus_score, analysis_data, timestamp) VALUES (?, ?, ?, ?)",
                (job_id, consensus_score, analysis_json, timestamp)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return False
    
    def get_interaction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the interaction history from the database, including associated responses.
        
        Args:
            limit (int): The maximum number of interactions to retrieve. Defaults to 50.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents an interaction.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT i.job_id, i.query, i.domain, i.question_type, i.timestamp, i.roles, a.consensus_score
            FROM interactions i
            LEFT JOIN analysis a ON i.job_id = a.job_id
            ORDER BY i.timestamp DESC
            LIMIT ?
            """, (limit,))
            
            interactions = []
            for row in cursor.fetchall():
                interaction = dict(row)
                
                # Get responses for this interaction
                resp_cursor = conn.cursor()
                resp_cursor.execute(
                    "SELECT agent_id, response, is_aggregator FROM responses WHERE job_id = ?",
                    (interaction['job_id'],)
                )
                
                responses = {}
                for resp_row in resp_cursor.fetchall():
                    responses[resp_row['agent_id']] = resp_row['response']
                
                interaction['responses'] = responses
                interactions.append(interaction)
            
            conn.close()
            return interactions
        except Exception as e:
            logger.error(f"Error getting interaction history: {str(e)}")
            return []
    
    def get_interaction(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves interaction details from the database for a given job ID.
        
        Args:
            job_id (str): The unique identifier for the job whose interaction details are to be retrieved.
        
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the interaction details if found, or None if no interaction
            exists for the given job ID.
        
        Raises:
            Logs an error and returns None if an exception occurs during database operations.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM interactions WHERE job_id = ?",
                (job_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            
            interaction = dict(row)
            
            # Get responses
            resp_cursor = conn.cursor()
            resp_cursor.execute(
                "SELECT agent_id, response, is_aggregator FROM responses WHERE job_id = ?",
                (job_id,)
            )
            
            responses = {}
            for resp_row in resp_cursor.fetchall():
                responses[resp_row['agent_id']] = {
                    'response': resp_row['response'],
                    'is_aggregator': bool(resp_row['is_aggregator'])
                }
            
            interaction['responses'] = responses
            
            # Get analysis
            analysis_cursor = conn.cursor()
            analysis_cursor.execute(
                "SELECT consensus_score, analysis_data FROM analysis WHERE job_id = ?",
                (job_id,)
            )
            
            analysis_row = analysis_cursor.fetchone()
            if analysis_row:
                interaction['consensus_score'] = analysis_row['consensus_score']
                interaction['analysis'] = json.loads(analysis_row['analysis_data'])
            
            conn.close()
            return interaction
        except Exception as e:
            logger.error(f"Error getting interaction: {str(e)}")
            return None
    
    def delete_interaction(self, job_id: str, username: str) -> bool:
        """
        Deletes an interaction and its associated data from the database.
        This method removes entries from the `responses`, `analysis`, and `interactions`
        tables in the database for a given job ID, provided the specified username
        matches the owner of the interaction.
        
        Args:
            job_id (str): The unique identifier of the job to be deleted.
            username (str): The username of the user attempting to delete the interaction.
        
        Returns:
            bool: True if the interaction and associated data were successfully deleted,
                  False if the username does not match the owner or an error occurs.
        
        Raises:
            None: Any exceptions encountered are logged and handled internally.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check ownership first
            cursor.execute("SELECT username FROM interactions WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row or row[0] != username:
                conn.close()
                return False
            
            # Delete from responses table first (foreign key constraint)
            cursor.execute("DELETE FROM responses WHERE job_id = ?", (job_id,))
            
            # Delete from analysis table
            cursor.execute("DELETE FROM analysis WHERE job_id = ?", (job_id,))
            
            # Delete from interactions table
            cursor.execute("DELETE FROM interactions WHERE job_id = ?", (job_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error deleting interaction: {str(e)}")
            return False

    def create_user(self, username: str, password: str) -> bool:
        """
        Creates a new user in the database with the given username and password.

        Args:
            username (str): The username of the new user.
            password (str): The plaintext password of the new user.

        Returns:
            bool: True if the user was successfully created.
            str: "duplicate" if the username already exists in the database.
            str: "error" if an unexpected error occurs during user creation.

        Raises:
            None: All exceptions are handled internally and logged.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            logger.warning("Attempted to create duplicate user.")
            return "duplicate"
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return "error"

    def verify_user(self, username: str, password: str) -> bool:
        """
        Verifies the credentials of a user by checking the provided username and password
        against the stored records in the database.

        Args:
            username (str): The username of the user to verify.
            password (str): The plaintext password of the user to verify.

        Returns:
            bool: True if the username and hashed password match a record in the database,
                  False otherwise.

        Raises:
            Exception: Logs an error message if an exception occurs during the verification process.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
            user = cursor.fetchone()
            conn.close()
            return user is not None
        except Exception as e:
            logger.error(f"Error verifying user: {str(e)}")
            return False

    def delete_user(self, username: str, password: str) -> bool:
        """
        Deletes a user from the database if the provided username and password match.
        The password is hashed using SHA-256.

        Args:
            username (str): The username of the user to be deleted.
            password (str): The plaintext password of the user to be deleted.

        Returns:
            bool: True if the user was successfully deleted, False otherwise.

        Raises:
            Exception: Logs an error if an exception occurs during the deletion process.          
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute("DELETE FROM users WHERE username = ? AND password = ?", (username, hashed_password))
            conn.commit()
            conn.close()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False
