'''
This file will contain utlity functions to handle data. It will include methods to read and write files.
'''
import os
import sqlite3
from configs.root_paths import *

class BasePromptDB:
    '''
    A class to manage the SQLite database for base prompts.
    This class is meant to handle all calls to the database. No other module anywhere will handle database calls directly. They will simply call the methods of this class.

    Further, this class will store the database connection object as an instance variable, so that the connection is not opened and closed for every call. This will improve performance. Every instance of this class will have a connection to the database, and the connection will be closed when the instance is destroyed.
    '''

    def __init__(self, db_path = SQL_DB):
        '''
        Initializes the BasePromptDB class and creates the database if it doesn't exist.

        Args:
            - db_path (str): The path to the SQLite database file.
        '''
        self.db_path = db_path
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        '''
        Initializes the database by creating the table if it doesn't exist.
        '''
        db_exists = os.path.exists(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        
        if not db_exists:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE base_prompts (
                    bpv_idx TEXT PRIMARY KEY,
                    base_prompt_string TEXT NOT NULL
                )
            ''')
            self.conn.commit()
            print(f"Database created at {self.db_path}")
        else:
            print(f"Database accessed at {self.db_path}")

    def insert_base_prompts(self, prompts):
        '''
        Inserts a batch of base prompts into the database.

        Args:
            - prompts (list of tuples): A list of tuples where each tuple contains (bpv_idx, base_prompt_string).
        '''
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO base_prompts (bpv_idx, base_prompt_string) VALUES (?, ?)
        ''', prompts)
        self.conn.commit()

    def fetch_all_prompts(self):
        '''
        Primarily used for testing purposes.
        Fetches all base prompts from the database.

        Returns:
            - list of tuples: A list of tuples where each tuple contains (bpv_idx, base_prompt_string).
        '''
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM base_prompts')
        return cursor.fetchall()
    
    def fetch_prompt(self, bpv_idx):
        '''
        Fetches a specific prompt from the database by bpv_idx.
        Is called by the Prompt class.

        Args:
            - bpv_idx (str): The base prompt variation index.

        Returns:
            - str: The prompt string.
        '''
        cursor = self.conn.cursor()
        cursor.execute('SELECT base_prompt_string FROM base_prompts WHERE bpv_idx = ?', (bpv_idx,))
        result = cursor.fetchone()
        return result[0] if result else None

    def close_connection(self):
        '''
        Closes the database connection.
        '''
        if self.conn:
            self.conn.close()
            print(f"Database connection to {self.db_path} closed.")

    def delete_database(self):
        '''
        Deletes the SQLite database file.
        Closes the connection before deleting the file.
        '''
        self.close_connection()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print(f"Database at {self.db_path} deleted.")
        else:
            print(f"No database found at {self.db_path} to delete.")

    def reset_database(self):
        '''
        Resets the database by deleting and recreating it.
        '''
        self.delete_database()
        self._initialize_db()


class PromptVariationParquet:
    '''
    Class used to manage the Prompt Variations in Parquet format as we discussed and is in README. This class will be initialized with a bpv_idx and will have methods to access, write, read prompt variations. It will be similar to the BasePromptDB class, but will have to handle different Parquet files indexed by base prompt, as opposed to a single SQLite database.
    '''


class ValidationScoreParquet:
    '''
    Class used to manage the Validation Scores in Parquet format as we discussed and is in README. This class will be initialized with a bpv_idx and will have methods to access, write, read validation scores. It will be very very similar to the PromptVariationParquet class. The only difference will be the data stored in the Parquet files.
    '''


class ModelOutputParquet:
    '''
    Class used to manage the Model Outputs in Parquet format as we discussed and is in README. This class will be initialized with a bpv_idx and will have methods to access, write, read model outputs. It will be very very similar to the PromptVariationParquet class but will have subfolders to iterate over and custom partitioned data. The data will include the learning guide.
    '''