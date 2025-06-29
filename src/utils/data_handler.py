'''
This file will contain utlity functions to handle data. It will include methods to read and write files.
'''
import os
import pandas as pd
import sqlite3
from configs.root_paths import *
import yaml
from configs.data_size_configs import *
import torch
import numpy as np
#NUM_RUBRIC_SECTIONS
#from prompt import ValidationScore

class PromptTopicDB:
    '''
    A class to manage the SQLite database for prompt topics.
    This class is meant to handle all calls to the database. No other module anywhere will handle database calls directly. They will simply call the methods of this class.

    Further, this class will store the database connection object as an instance variable, so that the connection is not opened and closed for every call. This will improve performance. Every instance of this class will have a connection to the database, and the connection will be closed when the instance is destroyed.
    '''

    def __init__(self, db_path = SQL_DB):
        '''
        Initializes the PromptTopicDB class and creates the database if it doesn't exist.

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
                CREATE TABLE topics (
                    idx INT PRIMARY KEY,
                    topic TEXT NOT NULL
                )
            ''')
            self.conn.commit()
            print(f"Database created at {self.db_path}")
        else:
            print(f"Database accessed at {self.db_path}")

    def insert_topics(self, prompts):
        '''
        Inserts a batch of topics into the database.

        Args:
            - prompts (list of tuples): A list of tuples where each tuple contains (idx, topic). Here idx is an integer and topic is a string.
        '''
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO topics (idx, topic) VALUES (?, ?)
        ''', prompts)
        self.conn.commit()

    def fetch_all_topics(self):
        '''
        Primarily used for testing purposes.
        Fetches all topics from the database.

        Returns:
            - list of strings: A list of strings, which are each topics.
        '''
        cursor = self.conn.cursor()
        cursor.execute('SELECT topic FROM topics')
        return cursor.fetchall()
    
    def fetch_topic(self, idx):
        '''
        Fetches a specific topic from the database by iidx.
        Is called by the Prompt class.

        Args:
            - idx (int): The index.

        Returns:
            - str: The topic string.
        '''
        cursor = self.conn.cursor()
        cursor.execute('SELECT topic FROM topics WHERE idx = ?', (idx,))
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
        print("Base prompts inserted.")

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

    def fetch_list_of_prompts(self, bp_idx_list):
        '''
        Fetches a list of prompts from the database by a list of bp_idx.

        Args:
            - bp_idx_list (list of int): A list of bp_idx.

        Returns:
            - list of str: A list of prompt strings.
        '''
        cursor = self.conn.cursor()
        placeholders = ','.join(['?'] * len(bp_idx_list))
        query = f'SELECT base_prompt_string FROM base_prompts WHERE bp_idx IN ({placeholders})'
        cursor.execute(query, bp_idx_list)
        results = cursor.fetchall()

        return [result[0] for result in results]
        

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

    def fetch_prompt_by_rownum(self, rownum):
        query = f"SELECT * FROM base_prompts LIMIT 1 OFFSET {rownum};"
        cursor = self.conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        if row:
            return row['base_prompt_string'], row['bp_idx']
        else:
            return None, None


class PromptVariationParquet:
    '''
    Class used to manage the Prompt Variations in Parquet format as we discussed and is in README. This class will be initialized with a bp_idx and will have methods to access, write, read prompt variations. It will be similar to the BasePromptDB class, but will have to handle different Parquet files indexed by base prompt, as opposed to a single SQLite database.
    '''

    def __init__(self, bp_idx, parquet_root_path = PROMPT_VARIATIONS, suffix=""):
        '''
        Initializes the PromptVariationParquet class with an optional suffix for the filename.
        '''
        self.parquet_root_path = parquet_root_path
        self.bp_idx = bp_idx
        self.suffix = suffix
        self.file_path = f'{self.parquet_root_path}/{bp_idx}_prompt_variation{self.suffix}.parquet'
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
        print(f'Variations in data_handler: {new_data}')
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
        self.df['bpv_idx'] = df['bpv_idx'].apply(tuple)
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
        self.df['bpv_idx'] = df['bpv_idx'].apply(tuple)
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
        self.df['bpv_idx'] = df['bpv_idx'].apply(tuple)
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
        self.file_path = f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'
        # self.df =  pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])
        # self._initialize_parquet() # Should this be initialize_parquet or access_parquet?
        self.df = self._access_parquet()

    def _initialize_parquet(self):
        # file_path = f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'
        # if not os.path.exists(file_path):
        #     self.df.to_parquet(file_path, index=False)
        #     print(f"Created new ValidationScoreParquet file at {file_path}")
        # else:
        #     self.df = pd.read_parquet(file_path)
        #     print(f"Using existing ValidationScoreParquet file at {file_path}")
        '''
        Initializes the Parquet file for the base prompt index if it doesn't exist. Otherwise, it does nothing.
        '''

        if not os.path.exists(self.file_path):
            df = pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + [f'section_{i+1}_avg' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])
            df.to_parquet(self.file_path, index=False)
            print(f"Created new ValidationScoreParquet file at {self.file_path}")
            # PRAVEEN: adding this line such that we store the updated df
            self.df = df

        else:
            print(f"Using existing ValidationScoreParquet file at {self.file_path}")

    def _access_parquet(self):
        '''
        An internal function that retrieves the content of the parquet file for a specific base prompt's prompt variation scores if it exists. Otherwise, it initializes the parquet file.
        
        Returns:
            - pd.DataFrame: A DataFrame containing all bpv_idx and their associated validation scores
        '''
        self._initialize_parquet()
        #return self.df
        return pd.read_parquet(self.file_path)

    # BECCA: Redundant function, may remove
    # def save_scores_to_parquet(self, scores_data):
    #     '''
    #     Saves the validation scores to the Parquet file for the base prompt.

    #     Args:
    #         - scores_list (list of tuples): A list of tuples where each tuple contains the bpv_idx and the scores for each rubric section and total score.
    #     '''
    #     if scores_data:
    #         self.insert_validation_scores([scores_data])

    # PRAVEEN: COMMENTING OUT THE FUNCTION BELOW BASED ON UPDATED RATIONALE - INSERTING AS A LIST OF VS_OBJs
    # def insert_validation_scores(self, scores_dict):
    #     '''
    #     Inserts a batch of validation scores into the respective parquet file. Assumes that the first model output of this batch is the base prompt model output.

    #     Args:
    #         - scores_dict (list of integers): A dict that contains the section scores, average section scores, and total score. This comes from ValidationScore.write_output().

    #     Raises:
    #         - ValueError: If the prompt variations are not specific to a single base prompt index.
    #     '''

    #     # Check if all validaiton scores are specific to a single base prompt index
    #     base_prompt_indexes = list(set([x[0][0] for x in scores_dict]))
    #     if len(base_prompt_indexes) > 1:
    #         raise ValueError("All validation scores must be specific to a single base prompt index.")
    #     if base_prompt_indexes[0] != self.bp_idx:
    #         raise ValueError("All validation scores must be specific to the base prompt index provided during initialization of the ValidationScoreParquet Object.")

    #     # CHANGE TO TAKE DICTIONARY (INSTEAD OF LIST) AS INPUT FOR CONVERSION INTO DATAFRAME
    #     df = pd.DataFrame(scores_dict, columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + [f'section_{i+1}_avg' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])
    #     # BECCA: Pending if the below seciton works as intended
    #     self.reset_parquet()
    #     df.to_parquet(self.file_path, index=False)

    #     # BECCA: Old last 2 lines below
    #     # self.df = pd.concat([self.df, new_data], ignore_index=True)
    #     # self.df.to_parquet(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet', index=False)

    # PRAVEEN: CREATING A NEW FUNCTION FOR INSERTING VS SCORES BASED ON UPDATED RATIONALE

    def insert_validation_scores(self, vs_objs_lst):
        '''
        For self.bp_idx, this method inserts validation scores for each prompt variation. 

        Args:
            vs_objs_lst: a list of vs_objs 
        '''
        # if there is already data, reset it
        self.reset_parquet()

        scores_list = []

        for vs_obj in vs_objs_lst:
            # retrieve the scores from the vs_obj
            scores_dict = vs_obj.get_scores()
            
            # add bpv_idx to the scores_dict
            scores_dict['bpv_idx'] = vs_obj.get_bpv_idx()

            scores_list.append(scores_dict)

        # convert the list of dictionaries to a DataFrame
        new_data = pd.DataFrame(scores_list, columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + [f'section_{i+1}_avg' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'])

        # update self.df with the new data
        self.df = pd.concat([self.df, new_data], ignore_index=True)

        # save the updated DataFrame to the Parquet file
        self.df.to_parquet(self.file_path, index=False)



    def fetch_all_validation_scores(self):
        '''
        Fetches all validation scores for the base prompt.

        Returns:
            - pd.DataFrame: A DataFrame containing all bpv_idx and their associated validation scores
        '''

        # if not os.path.exists(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'):
        #     return pd.DataFrame(columns = ['bpv_idx'] + [f'section_{i+1}' for i in range(NUM_RUBRIC_SECTIONS)] + ['total_score'], ignore_index=True)
        
        df = self._access_parquet()
        return df

    def fetch_all_agg_validation_scores(self):
        '''
        Fetches the aggregated validation scores for the base prompt. Returns the total score for each prompt variation.

        Returns:
            - pd.Series: A Series containing the total score for each prompt variation.

        '''
        df = self._access_parquet()
        return list(df['total_score'])

    def fetch_base_prompt_str(self):
        '''
        Returns the base prompt string associated with the bp_idx.
        Tries to fetch it from the PromptVariationParquet first, then falls back to BasePromptDB.
        '''
        try:
            # Attempt fetch from PromptVariationParqe
            prompt_variation_parquet = PromptVariationParquet(self.bp_idx, parquet_root_path = PROMPT_VARIATIONS)
            base_prompt_str = prompt_variation_parquet.fetch_base_prompt_str()

            if base_prompt_str:
                return base_prompt_str
        except Exception as e:
            print(f"Error fetching base prompt string from PromptVariationParquet: {e}")

        try:
            # If not found, fetch from BasePromptDB
            base_prompt_db = BasePromptDB(db_path = SQL_DB)
            base_prompt_str = base_prompt_db.fetch_prompt(self.bp_idx)

            return base_prompt_str if base_prompt_str else None  
        except Exception as e:
            print(f"Error fetching base prompt string from BasePromptDB: {e}")
            
        return None  
    
    def fetch_bpv_validation_scores(self, bpv_idx):
        '''
        Return the validation scores for the given bpv_idx.
        
        Inputs:
            - bpv_idx (tuple of ints): The base prompt variation index.

        Returns:
            - pd.DataFrame: A DataFrame containing the validation scores for the given bpv_idx.
        '''
        # BECCA: Old code, pending delete
        # if not os.path.exists(f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'):
            # return None
        # self._access_parquet()

        if bpv_idx[0] != self.bp_idx:
            raise ValueError("All model outputs must be specific to the base prompt index provided during initialization of the ValidatorScoreParquet Object.")
        result = self.df[self.df["bpv_idx"] == bpv_idx]
        # Drop the bpv_idx column and convert the remaining DataFrame to a dictionary
        result_dict = result.drop(columns=["bpv_idx"]).iloc[0].to_dict()
        
        return result_dict
    
    # BECCA: No longer needed? Pending removal
    # def fetch_prompt_variation_and_validation_scores(self, bpv_idx):
    #     '''
    #     Returns the prompt variation string and validation scores for the given bpv_idx.

    #     Inputs:
    #         - bpv_idx (tuple of ints): The base prompt variation index.
        
    #     Returns:
    #         - dict: A dictionary containing the prompt variation string and validation scores.
    #     '''
    #     prompt_variation = self.fetch_prompt_variation_str(bpv_idx)
    #     validation_scores = self.fetch_validation_scores(bpv_idx)
    #     return {"prompt_variation": prompt_variation, "validation_scores": validation_scores}

    def get_bp_idx(self):
        '''
        Returns the base prompt index associated with this object.
        '''
        return self.bp_idx
    
    def delete_parquet(self):
        '''
        Deletes the Parquet file associated with the base prompt.
        '''
        # BECCA: Old code, pending delete
        # file_path = f'{self.parquet_root_path}/{self.bp_idx}_validation_score.parquet'
        
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            print(f"Deleted ValidationScoreParquet file at {self.file_path}")
        else:
            print(f"No ValidationScoreParquet file found at {self.file_path} to delete.")
    
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

    def get_num_modeloutputs(self):
        num_mos = len(self.df) - 1
        return num_mos


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
        self.df['bpv_idx'] = self.df['bpv_idx'].apply(tuple)
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
        self.df['bpv_idx'] = self.df['bpv_idx'].apply(tuple)
        if bpv_idx[0] != self.bp_idx:
            return ValueError("All model outputs must be specific to the base prompt index provided during initialization of the ModelOutputParquet Object.")
        self.df['bpv_idx'] = self.df['bpv_idx'].apply(tuple)
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


class RegressionHeadDataset(torch.utils.data.Dataset):
    '''
    This class will be used to load and manage the dataset used for training the regression head. It will load all the prompt variations (including the base prompt) and their respective validation scores from the Parquet file structure.
    It will also shuffle the dataset and split it into training and validation sets.
    It assumes that all prompt variations are to be loaded at the same time.
    It is of type torch.utils.data.Dataset, so it can be used with DataLoader.
    '''

    def __init__(self, prompt_variations_path = PROMPT_VARIATIONS, validation_scores_path = VALIDATION_SCORES, regression_head_splits = REGRESSION_HEAD_CONFIG_PATH):
        '''
        Initializes the dataset with the path to the prompt variations and validation scores. Also creates the dataset and splits it into training, validation, and test sets. 
        '''
        self.prompt_variations_path = prompt_variations_path
        self.validation_scores_path = validation_scores_path
        self.regression_head_splits = regression_head_splits

        with open(self.regression_head_splits, 'r') as f:
            splits = yaml.safe_load(f)
            self.train_indices = splits['train_indices']
            self.test_indices = splits['test_indices']
            self.val_indices = splits['val_indices']

        # Load and merge the data
        self.train_data, self.test_data, self.val_data = self._load_and_merge_data()

        # Combine all data into a single DataFrame for indexing
        # self.data = pd.concat(self.train_data + self.test_data + self.val_data, ignore_index=True)
        self.data = self.train_data + self.test_data + self.val_data


    def _load_and_merge_data(self):
        '''
        Loads the parquet files for prompt variations and validation scores, and uses the training splits to create the train, test, val splits.
        '''
        # # Retrieve the prompt variations and validation scores files
        # prompt_variation_files = [os.path.join(self.prompt_variations_path, f) for f in os.listdir(self.prompt_variations_path) if f.endswith('.parquet')]
        # validation_score_files = [os.path.join(self.validation_scores_path, f) for f in os.listdir(self.validation_scores_path) if f.endswith('.parquet')]

        # # Sort the files by their numeric index extracted from the filenames
        # prompt_variation_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        # validation_score_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

        # # # For testing purposes, limit the number of files to 10
        # # prompt_variation_files = prompt_variation_files[:10]
        # # validation_score_files = validation_score_files[:10]

        # train_data = []
        # test_data = []
        # val_data = []

        # for pv_file, vs_file in zip(prompt_variation_files, validation_score_files):
        #     # Load Parquet files
        # Retrieve the prompt variations and validation scores files
        prompt_variation_files = [os.path.join(self.prompt_variations_path, f) for f in os.listdir(self.prompt_variations_path) if f.endswith('.parquet')]
        validation_score_files = [os.path.join(self.validation_scores_path, f) for f in os.listdir(self.validation_scores_path) if f.endswith('.parquet')]

        # Extract numeric IDs from filenames and create dictionaries for matching
        prompt_variation_dict = {int(os.path.basename(f).split('_')[0]): f for f in prompt_variation_files}
        validation_score_dict = {int(os.path.basename(f).split('_')[0]): f for f in validation_score_files}

        # Find common IDs between the two sets of files
        common_ids = set(prompt_variation_dict.keys()).intersection(validation_score_dict.keys())

        # Match files with the same ID
        matched_files = [(prompt_variation_dict[id], validation_score_dict[id]) for id in sorted(common_ids)]

        train_data = []
        test_data = []
        val_data = []

        for pv_file, vs_file in matched_files:
            prompt_variations = pd.read_parquet(pv_file)
            validation_scores = pd.read_parquet(vs_file)

            prompt_variations["bpv_idx"] = prompt_variations["bpv_idx"].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
            validation_scores["bpv_idx"] = validation_scores["bpv_idx"].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)

            # Merge on bpv_idx
            merged = pd.merge(prompt_variations, validation_scores, on="bpv_idx")

            if merged.empty:
                print(f"Warning: Merged DataFrame is empty for files {pv_file} and {vs_file}.")
                print(f'length of prompt variations: {len(prompt_variations)}')
                print(f'length of validation scores: {len(validation_scores)}')
                continue

            # Extract the prompt_variation_id from bpv_idx
            merged['prompt_variation_id'] = merged['bpv_idx'].apply(lambda x: x[1])
            base_prompt_str = merged.loc[merged['prompt_variation_id'] == -1, 'prompt_variation_string'].iloc[0]

            # Assign the base prompt string to all rows in the 'base_prompt' column
            merged['base_prompt'] = base_prompt_str

            # Convert the merged DataFrame into a list of tuples
            train_data.extend(
                merged.loc[merged['prompt_variation_id'].isin(self.train_indices)][
                    ['base_prompt', 'prompt_variation_string', 'total_score']
                ].itertuples(index=False, name=None)
            )
            test_data.extend(
                merged.loc[merged['prompt_variation_id'].isin(self.test_indices)][
                    ['base_prompt', 'prompt_variation_string', 'total_score']
                ].itertuples(index=False, name=None)
            )
            val_data.extend(
                merged.loc[merged['prompt_variation_id'].isin(self.val_indices)][
                    ['base_prompt', 'prompt_variation_string', 'total_score']
                ].itertuples(index=False, name=None)
            )

        return train_data, test_data, val_data
    
    def get_data_split(self):
        '''
        Returns the train, test, and validation splits of the dataset.
        '''
        return self.train_data, self.test_data, self.val_data
        
        

    # def _load_and_merge_data(self):
    #     '''
    #     Loads the parquet files for prompt variations and validation scores, and uses the training splits to create the train, test, val splits.
    #     '''
    #     # retrieve the prompt variations and validation scores files
    #     prompt_variation_files = [os.path.join(self.prompt_variations_path, f) for f in os.listdir(self.prompt_variations_path) if f.endswith('.parquet')]
    #     validation_score_files = [os.path.join(self.validation_scores_path, f) for f in os.listdir(self.validation_scores_path) if f.endswith('.parquet')]

    #     # Sort the files by their numeric index extracted from the filenames
    #     prompt_variation_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    #     validation_score_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

    #     # for each pair of prompt variations and validation scores files, load the data and merge them per split

    #     train_data = []
    #     test_data = []
    #     val_data = []

    #     for pv_file, vs_file in zip(prompt_variation_files, validation_score_files):
    #         # Load Parquet files
    #         prompt_variations = pd.read_parquet(pv_file)
    #         validation_scores = pd.read_parquet(vs_file)

    #         prompt_variations["bpv_idx"] = prompt_variations["bpv_idx"].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)
    #         validation_scores["bpv_idx"] = validation_scores["bpv_idx"].apply(lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x)

    #         # print(prompt_variations["bpv_idx"].head())
    #         # print(type(prompt_variations["bpv_idx"].iloc[0]))

    #         # print(validation_scores["bpv_idx"].head())
    #         # print(type(validation_scores["bpv_idx"].iloc[0]))

    #         # Merge on bpv_idx
    #         merged = pd.merge(prompt_variations, validation_scores, on="bpv_idx")
    #         # print(merged['bpv_idx'].values)

    #         # Extract the prompt_variation_id from bpv_idx
    #         merged['prompt_variation_id'] = merged['bpv_idx'].apply(lambda x: x[1])
    #         # base_prompt_str = merged[merged['prompt_variation_id'] == -1]['prompt_variation_string']
    #         # print(base_prompt_str)
    #         # merged['base_prompt'] = base_prompt_str
    #         base_prompt_str = merged.loc[merged['prompt_variation_id'] == -1, 'prompt_variation_string'].iloc[0]

    #         # Assign the base prompt string to all rows in the 'base_prompt' column
    #         merged['base_prompt'] = base_prompt_str

    #         # Filter the data based on train, test, and val indices
    #         train_data.append(merged.loc[merged['prompt_variation_id'].isin(self.train_indices)])
    #         test_data.append(merged.loc[merged['prompt_variation_id'].isin(self.test_indices)])
    #         val_data.append(merged.loc[merged['prompt_variation_id'].isin(self.val_indices)])
        
    #     return train_data, test_data, val_data


    def get_subset(self, split):
        '''
        Returns a subset of the dataset (train, test, or validation).

        Args:
            split (str): The subset to return ('train', 'test', or 'val').

        Returns:
            RegressionHeadDatasetSubset: A subset of the dataset.
        '''
        if split == "train":
            indices = self.train_indices
        elif split == "test":
            indices = self.test_indices
        elif split == "val":
            indices = self.val_indices
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'val'.")

        return RegressionHeadDatasetSubset(self.data, indices)
    
    def __len__(self):
        '''
        Returns the total number of samples in the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns a single data sample for the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the base prompt, prompt variation, and total score.
        '''
        row = self.data.iloc[idx]
        return {
            "base_prompt": row["base_prompt"],
            "prompt_variation": row["prompt_variation_string"],
            "total_score": row["total_score"]
        }
    
class RegressionHeadDatasetSubset(torch.utils.data.Dataset):
    '''
    A subset of the RegressionHeadDataset (train, test, or validation).
    '''

    def __init__(self, full_data, indices):
        '''
        Args:
            full_data (pd.DataFrame): The full dataset loaded by the master class.
            indices (list): Indices for the subset.
        '''
        self.data = full_data.loc[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "base_prompt": row["base_prompt"],
            "prompt_variation": row["prompt_variation_string"],
            "total_score": row["total_score"]
        }

# if __name__ == "__main__":
#     dataset = RegressionHeadDataset()
#     train = dataset.train_data
#     test = dataset.test_data
#     val = dataset.val_data

#     print("Train Data:", train)
#     print("Test Data:", test)
#     print("Validation Data:", val)


if __name__ == "__main__":
    bp_db = BasePromptDB()
    test_indices = [0, 1, 2]
    test_results = bp_db.fetch_list_of_prompts(test_indices)
    print("Test Results:", test_results)
