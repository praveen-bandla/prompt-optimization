'''
This file will contain utlity functions to handle data. It will include methods to read and write files.
'''
import os
import pandas as pd
import sqlite3
from configs.root_paths import *
from configs.data_size_configs import *
#NUM_RUBRIC_SECTIONS
#from prompt import ValidationScore

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
            # cursor.execute('''
            #     CREATE TABLE base_prompts (
            #         bpv_idx TEXT PRIMARY KEY,
            #         base_prompt_string TEXT NOT NULL
            #     )
            # ''')
            cursor.execute('''
                CREATE TABLE base_prompts (
                    bp_idx INT PRIMARY KEY,
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
            - prompts (list of tuples): A list of tuples where each tuple contains (bp_idx, base_prompt_string). Here bp_idx is an integer and base_prompt_string is a string.
        '''
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO base_prompts (bp_idx, base_prompt_string) VALUES (?, ?)
        ''', prompts)
        self.conn.commit()

    def fetch_all_prompts(self):
        '''
        Primarily used for testing purposes.
        Fetches all base prompts from the database.

        Returns:
            - list of tuples: A list of tuples where each tuple contains (bp_idx, base_prompt_string).
        '''
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM base_prompts')
        return cursor.fetchall()
    
    def fetch_prompt(self, bp_idx):
        '''
        Fetches a specific prompt from the database by bp_idx.
        Is called by the Prompt class.

        Args:
            - bp_idx (int): The base prompt variation index.

        Returns:
            - str: The prompt string.
        '''
        cursor = self.conn.cursor()
        cursor.execute('SELECT base_prompt_string FROM base_prompts WHERE bp_idx = ?', (bp_idx,))
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
    Class used to manage the Prompt Variations in Parquet format as we discussed and is in README. This class will be initialized with a bp_idx and will have methods to access, write, read prompt variations. It will be similar to the BasePromptDB class, but will have to handle different Parquet files indexed by base prompt, as opposed to a single SQLite database.
    '''

    def __init__(self, bp_idx, parquet_root_path = PROMPT_VARIATIONS):
        '''
        Initializes the PromptVariationParquet class. This is base prompt specific. When needing to access prompt variations for a specific base prompt, the bp_idx will be passed to the methods of this class.
        '''
        self.parquet_root_path = parquet_root_path
        self.bp_idx = bp_idx
        self.file_path = f'{self.parquet_root_path}/{bp_idx}_prompt_variation.parquet'
        self.df = self._access_parquet()

    def _initialize_parquet(self): 
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=["bpv_idx", "prompt_variation_string"])
            df.to_parquet(self.file_path, index=False)
            print(f"Created new Parquet file at {self.file_path}")
        else:
            print(f"Using existing Parquet file at {self.file_path}")

    def _access_parquet(self):
        '''
        An internal function that retrieves the content of the parquet file for a specific base prompt index if it exists. Otherwise, it initializes the parquet file.
        '''
        self._initialize_parquet()
        return pd.read_parquet(self.file_path)

    def insert_prompt_variations(self,variations):
        '''
        Inserts a batch of prompt variations into the respective parquet file. If prompt variations already exist, this function resets the parquet file and inserts the new prompt variations.

        Args:
            - variations (list of tuples): A list of tuples where each tuple contains (bpv_idx, prompt_variation_string). THESE HAVE TO BE SPECIFIC TO A BASE PROMPT INDEX. CANNOT MIX PROMPT VARIATIONS FOR DIFFERENT BASE PROMPTS.
        '''
        
        # df = self._access_parquet(base_prompt_indexes[0])
        self.reset_parquet()
        new_data = pd.DataFrame(variations, columns=["bpv_idx", "prompt_variation_string"])
        df = pd.concat([self.df, new_data], ignore_index=True)
        df.to_parquet(self.file_path, index=False)

    def fetch_all_variations(self):
        '''
        Fetches all prompt variations for the base prompt.
        
        Returns:
            - List 1: all bpv_idx
            - List 2: all prompt_variation_strings 
        '''
        df = self._access_parquet()
        return df["bpv_idx"], df["prompt_variation_string"]
    
    def fetch_base_prompt_str(self):
        '''
        For a given bpv_idx, fetches the base prompt string associated with the base_prompt_idx.
        '''
        result = self.df[self.df["bpv_idx"] == (self.bp_idx, -1)]['prompt_variation_string']
        return result.iloc[0] if not result.empty else None

    def fetch_prompt_variation(self, bpv_idx):
        '''
        Returns the prompt variation string for the given bpv_idx.

        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.

        Returns:
            - str: The prompt variation string if found, else None.
        '''
        if bpv_idx[0] != self.bp_idx:
            return ValueError("All model outputs must be specific to the base prompt index provided during initialization of the PromptVariationParquet Object.")
        result = self.df[self.df["bpv_idx"] == bpv_idx]['prompt_variation_string']
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
        base_prompt = self.df[self.df["bpv_idx"] == (self.bp_idx, -1)]['prompt_variation_string']
        prompt_variation = self.df[self.df["bpv_idx"] == bpv_idx]['prompt_variation_string']
        return base_prompt.iloc[0] if not base_prompt.empty else None, prompt_variation.iloc[0] if not prompt_variation.empty else None


    def delete_parquet(self):
        '''
        Deletes the Parquet file associated with the base prompt.
        '''
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"Deleted Parquet file at {self.file_path}")
        else:
            print(f"No Parquet file found at {self.file_path} to delete.")
    
    def reset_parquet(self):
        '''
        Resets the Parquet file associated with the provided bp_idx by deleting and recreating it.
        '''
        self.delete_parquet()
        self._initialize_parquet()
        self.df = self._access_parquet() 


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
        # file_path = f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'
        # if not os.path.exists(file_path):
        #     self.df.to_parquet(file_path, index=False)
        #     print(f"Created new ValidationScoreParquet file at {file_path}")
        # else:
        #     self.df = pd.read_parquet(file_path)
        #     print(f"Using existing ValidationScoreParquet file at {file_path}")

        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + [f'section_{i+1}_avg' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])
            df.to_parquet(self.file_path, index=False)
            print(f"Created new ValidationScoreParquet file at {self.file_path}")

        else:
            print(f"Using existing ValidationScoreParquet file at {self.file_path}")

    def _access_parquet(self):
        '''
        An internal function that retrieves the content of the parquet file for a specific base prompt's prompt variation scores if it exists. Otherwise, it initializes the parquet file.
        '''
        self._initialize_parquet()
        #return self.df
        return pd.read_parquet()

    def save_scores_to_parquet(self, scores_data):
        '''
        Saves the validation scores to the Parquet file for the base prompt.

        Args:
            - scores_list (list of tuples): A list of tuples where each tuple contains the bpv_idx and the scores for each rubric section and total score.
        '''
        if scores_data:
            self.insert_validation_scores([scores_data])

    def insert_validation_scores(self, scores_list):
        '''
        Inserts ALL validation scores into the respective parquet file.

        Args:
            - scores_list (list of integers): A list that contains the scores, average section scores, and total score.
        '''

        new_data = pd.DataFrame(scores_list, columns=["bpv_idx"] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ["total_score"], ignore_index=False)
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self.df.to_parquet(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet', index=False)

    def fetch_all_validation_scores(self):
        '''
        Fetches all validation scores for the base prompt.

        Returns:
            - pd.DataFrame: A DataFrame containing all bpv_idx and their associated validation scores
        '''

        if not os.path.exists(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'):
            return pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'], ignore_index=True)
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
    Class used to manage the Model Outputs in Parquet format as we discussed and is in README. This class will be initialized with a bp_idx and will have methods to access, write, read model outputs. It will be very very similar to the PromptVariationParquet class. For our first task, the model output is equivalent to the learning guide.
    '''
    def __init__(self, bp_idx, parquet_root_path = MODEL_OUTPUTS):
        '''
        Initializes the ModelOutputParquet class. It will be used to manage all base prompt and base prompt variation outputs in the project. When needing to access prompt outputs for a specific prompt, the bpv_idx will be passed to the methods of this class.

        Args:
            - bp_idx (int): The base prompt index.
            - parquet_root_path (str): The path to the root directory where the Parquet files are stored.
        '''
        self.parquet_root_path = parquet_root_path
        self.bp_idx = bp_idx
        self.file_path = f'{self.parquet_root_path}/{self.bp_idx}_model_output.parquet'
        self.df = self._access_parquet()

    def _initialize_parquet(self):
        '''
        Initializes the Parquet file for the base prompt index if it doesn't exist. Otherwise, it does nothing.
        '''
        #file_path = f'{self.parquet_root_path}/{self.bp_idx}_model_output.parquet'
        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns=["bpv_idx", "model_output_string"])
            df.to_parquet(self.file_path, index=False)
            print(f"Created new Parquet file at {self.file_path}")
        else:
            print(f"Using existing Parquet file at {self.file_path}")

    def _access_parquet(self):
        '''
        An internal function that retrieves the content of the parquet file for a specific base prompt index.
        '''
        self._initialize_parquet()
        return pd.read_parquet(self.file_path)

    def insert_model_outputs(self, model_outputs):
        '''
        Inserts a batch of model outputs into the respective parquet file. Assumes that the first model output of this batch is the base prompt model output.

        Args:
            - model outputs (list of tuples): A list of tuples where each tuple contains (bpv_idx, model_output_string). THESE HAVE TO BE SPECIFIC TO A BASE PROMPT INDEX. CANNOT MIX PROMPT VARIATIONS FOR DIFFERENT BASE PROMPTS. 

        Raises:
            - ValueError: If the prompt variations are not specific to a single base prompt index.
        '''
        # Check if all model outputs are specific to a single base prompt index
        base_prompt_indexes = list(set([x[0][0] for x in model_outputs]))
        if len(base_prompt_indexes) > 1:
            raise ValueError("All model outputs must be specific to a single base prompt index.")
        if base_prompt_indexes[0] != self.bp_idx:
            raise ValueError("All model outputs must be specific to the base prompt index provided during initialization of the ModelOutputParquet Object.")
        
        # df = self._access_parquet(self.bp_idx)

        # new_data = pd.DataFrame(model_outputs, columns=["bpv_idx", "model_output_string"])
        # df = pd.concat([self.df, new_data], ignore_index=True)
        print(model_outputs)
        df = pd.DataFrame(model_outputs, columns=["bpv_idx", "model_output_string"])
        self.reset_parquet()
        df.to_parquet(self.file_path, index=False)

    def fetch_all_outputs(self):
        '''
        Fetches all model outputs for a given base prompt.

        Returns:
            - List 1: all bpv_idx
            - List 2: all model_output_strings
        '''
        df = self._access_parquet()
        return df["bpv_idx"], df["model_output_string"]
    
    def fetch_base_prompt_str(self):
        '''
        Fetches the base prompt string associated with the base_prompt_idx.
        '''
        result = self.df[self.df["bpv_idx"] == (self.bp_idx, -1)]['model_output_string']
        return result.iloc[0] if not result.empty else None

    def fetch_model_output(self, bpv_idx):
        '''
        Returns the model output string for the given bpv_idx.

        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.

        Returns:
            - str: The model output string if found, else None.
        '''
        if bpv_idx[0] != self.bp_idx:
            return ValueError("All model outputs must be specific to the base prompt index provided during initialization of the ModelOutputParquet Object.")
        result = self.df[self.df["bpv_idx"] == bpv_idx]['model_output_string']
        return result.iloc[0] if not result.empty else None
    
    def get_bp_idx(self):
        '''
        Returns the base prompt index associated with this object.
        '''
        return self.bp_idx
    
    # no longer needed but praveen said leave it here so i listen like an obedient child
    # def fetch_base_prompt_and_model_output(self, bpv_idx):
    #     '''
    #     Returns the base prompt string and prompt variation string for the given bpv_idx.

    #     Inputs:
    #         - bpv_idx (tuple of ints): The base prompt variation index.
        
    #     Returns:
    #         - str: The base prompt string if found, else None.
    #         - str: The model output string if found, else None.
    #     '''
    #     if bpv_idx[0] != self.bp_idx:
    #         return ValueError("All model outputs must be specific to the base prompt index provided during initialization of the ModelOutputParquet Object.")
    #     base_prompt = self.df[self.df["bpv_idx"] == (self.bp_idx, -1)]['model_output_string']
    #     model_output = self.df[self.df["bpv_idx"] == bpv_idx]['model_output_string']
    #     return base_prompt.iloc[0] if not base_prompt.empty else None, model_output.iloc[0] if not model_output.empty else None

    def delete_parquet(self):
        '''
        Deletes the Parquet file associated with the base prompt.
        '''
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"Deleted Parquet file at {self.file_path}")
        else:
            print(f"No Parquet file found at {self.file_path} to delete.")
    
    def reset_parquet(self):
        '''
        Resets the Parquet file associated with the provided bp_idx by deleting and recreating it.
        '''
        self.delete_parquet()
        self._initialize_parquet()
