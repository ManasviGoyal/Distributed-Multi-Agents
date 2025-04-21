import os
import sqlite3
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path = os.path.join(os.path.dirname(__file__), "../agent_history.db")):
        self.db_path = db_path
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize the SQLite database with necessary tables"""
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
                username TEXT NOT NULL
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
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
            SELECT i.job_id, i.query, i.domain, i.question_type, i.timestamp, a.consensus_score
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

    def save_interaction(self, job_id: str, query: str, domain: str, question_type: str, username: str) -> bool:
        """Save a new interaction to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = int(time.time())
            
            cursor.execute(
                "INSERT INTO interactions (job_id, query, domain, question_type, timestamp, username) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, query, domain, question_type, timestamp, username)
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving interaction: {str(e)}")
            return False
    
    def save_responses(self, job_id: str, responses: Dict[str, str], aggregator_id: Optional[str] = None) -> bool:
        """Save model responses to the database"""
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
        """Save analysis results to the database"""
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
        """Get the interaction history with associated responses"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT i.job_id, i.query, i.domain, i.question_type, i.timestamp,
                   a.consensus_score
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
        """Get a specific interaction by job_id"""
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
        """Delete an interaction and its associated data"""
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
