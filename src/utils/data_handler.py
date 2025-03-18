'''
This file will contain utlity functions to handle data. It will include methods to read and write files.
'''
import os
import sqlite3
from configs.root_paths import *
import pandas as pd

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
    Class used to manage the Prompt Variations in Parquet format as we discussed and is in README. This class will be initialized with a base prompt index and will have methods to access, write, read prompt variations. It will be similar to the BasePromptDB class, but will have to handle different Parquet files indexed by base prompt, as opposed to a single SQLite database.
    '''
    # number of variations, base prompts should be called from a config file
    # def __init__(self, base_prompt_idx):
    #     '''
    #     Initializes the PromptVariationParquet class for a specific base prompt.
        
    #     Args:
    #         - base_prompt_idx (int): The index of the base prompt.
    #     '''
    #     self.base_prompt_idx = base_prompt_idx
    #     self.file_path = Path("data/base_prompts") / f"{base_prompt_idx}_pv.parquet"
    #     self._initialize_parquet()

    def __init__(self, parquet_root_path = PROMPT_VARIATIONS):
        '''
        Initializes the PromptVariationParquet class. This is not base prompt specific. It will be used to manage all prompt variations in the project. When needing to access prompt variations for a specific base prompt, the bpv_idx will be passed to the methods of this class.
        '''
        self.parquet_root_path = parquet_root_path
    
    # def _initialize_parquet(self):
    #     '''
    #     Initializes the Parquet file if it does not exist.
    #     '''
    #     if not self.file_path.exists():
    #         df = pd.DataFrame(columns=["bpv_idx", "prompt_variation_string"])
    #         df.to_parquet(self.file_path, index=False)
    #         print(f"Created new Parquet file at {self.file_path}")
    #     else:
    #         print(f"Using existing Parquet file at {self.file_path}")

    def _initialize_parquet(self, bp_idx):
        file_path = f'{self.parquet_root_path}/{bp_idx}_pv.parquet'
        if not os.path.exists(file_path):
            df = pd.DataFrame(columns=["bpv_idx", "prompt_variation_string"])
            df.to_parquet(file_path, index=False)
            print(f"Created new Parquet file at {file_path}")
        else:
            print(f"Using existing Parquet file at {file_path}")
    
    # def insert_prompt_variations(self, variations):
    #     '''
    #     Inserts a batch of prompt variations into the Parquet file.
        
    #     Args:
    #         - variations (list of tuples): A list of tuples where each tuple contains (bpv_idx, prompt_variation_string).
    #     '''
    #     df = pd.read_parquet(self.file_path)
    #     new_data = pd.DataFrame(variations, columns=["bpv_idx", "prompt_variation_string"])
    #     df = pd.concat([df, new_data], ignore_index=True)
    #     df.to_parquet(self.file_path, index=False)

    def _access_parquet(self, bp_idx):
        '''
        An internal function that retrieves the content of the parquet file for a specific base prompt index if it exists. Otherwise, it initializes the parquet file.
        '''
        self._initialize_parquet(bp_idx)
        return pd.read_parquet(f'{self.parquet_root_path}/{bp_idx}_pv.parquet')

    def insert_prompt_variations(self,variations):
        '''
        Inserts a batch of prompt variations into the respective parquet file.

        Args:
            - variations (list of tuples): A list of tuples where each tuple contains (bpv_idx, prompt_variation_string). THESE HAVE TO BE SPECIFIC TO A BASE PROMPT INDEX. CANNOT MIX PROMPT VARIATIONS FOR DIFFERENT BASE PROMPTS.

        Raises:
            - ValueError: If the prompt variations are not specific to a single base prompt index.
        '''
        base_prompt_indexes = list(set([x[0] for x in variations]))
        if len(base_prompt_indexes) > 1:
            raise ValueError("All prompt variations must be specific to a single base prompt index.")
        
        df = self._access_parquet(base_prompt_indexes[0])

        new_data = pd.DataFrame(variations, columns=["bpv_idx", "prompt_variation_string"])
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_parquet(f'{self.parquet_root_path}/{base_prompt_indexes[0]}_pv.parquet', index=False)

    def fetch_all_variations(self, bp_idx):
        '''
        Fetches all prompt variations for the base prompt.

        Inputs:
            - bp_idx (int): The base prompt index.
        
        Returns:
            - List 1: all bpv_idx
            - List 2: all prompt_variation_strings 
        '''

        if not os.path.exists(f'{self.parquet_root_path}/{bp_idx}_pv.parquet'):
            return [], []
        df = self._access_parquet(bp_idx)
        return df["bpv_idx"], df["prompt_variation_string"]
    
    def fetch_base_prompt_str(self, bpv_idx):
        '''
        For a given bpv_idx, fetches the base prompt string associated with the base_prompt_idx.
        '''
        bp_idx = bpv_idx[0]
        if not os.path.exists(f'{self.parquet_root_path}/{bp_idx}_pv.parquet'):
            return None
        df = self._access_parquet(bp_idx)
        result = df[df["bpv_idx"] == (bp_idx, -1)]['prompt_variation_string']
        return result.iloc[0] if not result.empty else None

    def fetch_prompt_variation(self, bpv_idx):
        '''
        Returns the prompt variation string for the given bpv_idx.

        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.

        Returns:
            - str: The prompt variation string if found, else None.
        '''
        bp_idx = bpv_idx[0]
        df = self._access_parquet(bp_idx)
        result = df[df["bpv_idx"] == bpv_idx]['prompt_variation_string']
        return result.iloc[0] if not result.empty else None
    
    def fetch_base_prompt_and_prompt_variation(self, bpv_idx):
        '''
        Returns the base prompt string and prompt variation string for the given bpv_idx.

        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.
        
        Returns:
            - str: The base prompt string if found, else None.
            - str: The prompt variation string if found, else None.
        '''

        bp_idx = bpv_idx[0]
        df = self._access_parquet(bp_idx)
        base_prompt = df[df["bpv_idx"] == (bp_idx, -1)]['prompt_variation_string']
        prompt_variation = df[df["bpv_idx"] == bpv_idx]['prompt_variation_string']
        return base_prompt.iloc[0] if not base_prompt.empty else None, prompt_variation.iloc[0] if not prompt_variation.empty else None


    def delete_parquet(self, bp_idx):
        '''
        Deletes the Parquet file associated with the base prompt.
        Inputs:
            - bp_idx (int): The base prompt index.
        '''
        file_path = f'{self.parquet_root_path}/{bp_idx}_pv.parquet'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted Parquet file at {file_path}")
        else:
            print(f"No Parquet file found at {file_path} to delete.")
    
    def reset_parquet(self, bp_idx):
        '''
        Resets the Parquet file associated with the provided bp_idx by deleting and recreating it.
        Inputs:
            - bp_idx (int): The base prompt index.
        '''
        self.delete_parquet(bp_idx)
        self._initialize_parquet(bp_idx) 


class ValidationScoreParquet:
    '''
    Class used to manage the Validation Scores in Parquet format as we discussed and is in README. This class will be initialized with a base prompt index and will have methods to access, write, read validation scores. It will be very very similar to the PromptVariationParquet class. The only difference will be the data stored in the Parquet files.
    '''


class ModelOutputParquet:
    '''
    Class used to manage the Model Outputs in Parquet format as we discussed and is in README. This class will be initialized with a bpv_idx and will have methods to access, write, read model outputs. It will be very very similar to the PromptVariationParquet class but will have subfolders to iterate over and custom partitioned data. For our first task, the model output is equivalent to the learning guide.
    '''