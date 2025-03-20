'''
This file will contain utlity functions to handle data. It will include methods to read and write files.
'''
import os
import pandas as pd
import sqlite3
from configs.root_paths import *
from configs.data_size_configs import NUM_RUBRIC_SECTIONS
from prompt import ValidationScore

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
    Class used to manage the Validation Scores in Parquet format as we discussed and is in README. 
    This class will be initialized with a bp_idx and will have methods to access, write, read validation scores. 
    It will be very very similar to the PromptVariationParquet class. 
    The only difference will be the data stored in the Parquet files.
    '''

    '''
    Class used to manage the Prompt Variations in Parquet format as we discussed and is in README. This class will be initialized with a base prompt index and will have methods to access, write, read prompt variations. It will be similar to the BasePromptDB class, but will have to handle different Parquet files indexed by base prompt, as opposed to a single SQLite database.
    '''

    def __init__(self, bp_idx, parquet_root_path = VALIDATION_SCORES): 
        '''
        Initializes the ValidationScoreParquet class. This is not base prompt specific. 
        It will be used to manage all validation scores in the project. 
        When needing to access validation scores for a specific base prompt, the bpv_idx will be passed to the methods of this class.
        '''
        self.bp_idx = bp_idx
        self.parquet_root_path = parquet_root_path
        self.df =  pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])
        self._initialize_parquet()

    def _initialize_parquet(self):
        file_path = f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'
        if not os.path.exists(file_path):
            self.df.to_parquet(file_path, index=False)
            print(f"Created new ValidationScoreParquet file at {file_path}")
        else:
            self.df = pd.read_parquet(file_path)
            print(f"Using existing ValidationScoreParquet file at {file_path}")

    def save_scores_to_parquet(self, scores_data):
        '''
        Saves the validation scores to the Parquet file for the base prompt.

        Args:
            - scores_list (list of tuples): A list of tuples where each tuple contains the bpv_idx and the scores for each rubric section and total score.
        '''
        if scores_data:
            self.insert_validation_scores([scores_data])


    def _access_parquet(self):
        '''
        An internal function that retrieves the content of the parquet file for a specific base prompt's prompt variation scores if it exists. Otherwise, it initializes the parquet file.
        '''
        self._initialize_parquet()
        return self.df

    def insert_validation_scores(self, scores_list):
        '''
        Inserts ALL validation scores into the respective parquet file - no batched writing.

        Args:
            - scores_list (list of integers): A list that contains the scores, average section scores, and total score.
        '''

        new_data = pd.DataFrame(scores_list, columns=["bpv_idx"] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ["total_score"])
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self.df.to_parquet(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet', index=False)

    def fetch_all_validation_scores(self):
        '''
        Fetches all validation scores for the base prompt.

        Returns:
            - pd.DataFrame: A DataFrame containing all bpv_idx and their associated validation scores
        '''

        if not os.path.exists(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'):
            return pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])
        self._access_parquet()
        return self.df

    def fetch_prompt_variation_str(self, bpv_idx):
        '''
        Returns the prompt variation string for the given bpv_idx.

        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.

        Returns:
            - str: The prompt variation string if found, else None.
        '''
        prompt_variation_parquet = PromptVariationParquet()
        df = prompt_variation_parquet._access_parquet(self.bp_idx)
        result = df[df["bpv_idx"] == bpv_idx]['prompt_variation_string']
        return result.iloc[0] if not result.empty else None
    
    def fetch_validation_scores(self, bpv_idx):
        """For a given bpv_idx, fetches the validation scores associated with it."""
        if not os.path.exists(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'):
            return None
        self._access_parquet()
        result = self.df[self.df["bpv_idx"] == bpv_idx]
        return result if not result.empty else None
    
    def fetch_prompt_variation_and_validation_scores(self, bpv_idx):
        '''
        Returns the prompt variation string and validation scores for the given bpv_idx.

        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.
        
        Returns:
            - dict: A dictionary containing the prompt variation string and validation scores.
        '''
        prompt_variation = self.fetch_prompt_variation_str(bpv_idx)
        validation_scores = self.fetch_validation_scores(bpv_idx)
        return {"prompt_variation": prompt_variation, "validation_scores": validation_scores}

    def delete_parquet(self):
        '''
        Deletes the Parquet file associated with the base prompt.
        '''
        file_path = f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted ValidationScoreParquet file at {file_path}")
        else:
            print(f"No ValidationScoreParquet file found at {file_path} to delete.")
    
    def reset_parquet(self):
        '''
        Resets the Parquet file associated with the object's bp_idx by deleting and recreating it.
        Inputs:
            - bp_idx (int): The base prompt index.
        '''
        self.delete_parquet()
        self._initialize_parquet() 

class ModelOutputParquet:
    '''
    Class used to manage the Model Outputs in Parquet format as we discussed and is in README. This class will be initialized with a bpv_idx and will have methods to access, write, read model outputs. It will be very very similar to the PromptVariationParquet class but will have subfolders to iterate over and custom partitioned data. The data will include the learning guide.
    '''