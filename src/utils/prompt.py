'''
This class will be used to manage Prompts. It will include methods to create prompt, access prompts, write prompts, read prompts.
'''

from utils.data_handler import BasePromptDB
from configs.root_paths import *
from utils.data_handler import * 
import pandas as pd

class BasePrompt:
    '''
    This class will be used to manage Base Prompts.
    It will include methods to create base prompt, access base prompt, write base prompt, read base prompt. Every component of the project that needs to access base prompts will use this class exclusively. 
    '''

    def __init__(self, bpv_idx):
        '''
        Initialize a base prompt with a base prompt variation index (given -1 for the second index of the tuple). Whether to create a new base prompt, or access an existing base prompt, this base prompt class will be used.

        Args:
            - bpv_idx (tuple of ints): The base prompt variation index.
        '''
        self.bpv_idx = bpv_idx
        self.prompt = None

    def get_prompt_str(self, db = None):
        '''
        Fetches the prompt from the database.
        If a database connection is not provided, a new connection is created and closed. If it is provided, it is used and not closed.

        Args:
            - db (BasePromptDB, optional): An existing database connection object.

        Returns:
            - str: The prompt string.
        '''
        if db is None:
            db = BasePromptDB()
            self.prompt = db.fetch_prompt(self.bpv_idx)
            db.close_connection()
        else:
            self.prompt = db.fetch_prompt(self.bpv_idx)
        
        return self.prompt
    
    def get_prompt_index(self):
        '''
        Returns the base prompt variation index.

        Returns:
            - tuple of ints: The base prompt variation index.
        '''
        return self.bpv_idx
    
    def save_base_prompt(self, db = None):
        '''
        Saves the base prompt to the database.
        If a database connection is not provided, a new connection is created and closed. If it is provided, it is used and not closed.

        Args:
            - db (BasePromptDB, optional): An existing database connection object.
        '''
        if db is None:
            db = BasePromptDB()
            db.insert_base_prompts([(self.bpv_idx, self.prompt)])
            db.close_connection()
        else:
            db.insert_base_prompts([(self.bpv_idx, self.prompt)])


class PromptVariation:
    '''
    Class to be developed that will be used to manage Prompt Variations in a similar way BasePrompt manages base prompts. Will be initialized with a bpv_idx and will have methods to access, write, read prompt variations, just like above.
    '''

class ValidationScore:
    '''
    Class to be developed that will be used to manage Validation Scores. Will include methods to create validation score, access validation score, write validation score, read validation score, just like PromptVariation and BasePrompt. This class will also have to include methods to calculate the aggregated scores, per category score, etc.
    '''


class MainModelOutput:
    '''
    Class to be developed that will be used to manage the main model outputs. Will include methods to create main model output, access main model output, write main model output, read main model output, just like PromptVariation and BasePrompt. This class will be most similar to PromptVariation.
    We are assuming that a ModelOutputParquet object has been created before an instantiation of this class is created.
    '''

    def __init__(self, bpv_idx, mo_parquet, model_output_str=None):
        '''
        Initialize a main model output with a base prompt variation index.

        Args:
            - bpv_idx (tuple of ints): The base prompt variation index.
            - mo_parquet (ModelOutputParquet): The Parquet database object.
            - model_output_str (str, optional): The main model output string
        '''
        self.bpv_idx = bpv_idx
        self.bp = bpv_idx[0]
        self.model_output_str = model_output_str
        self.mo_parquet = mo_parquet
    
    def get_output(self):
        '''
        Fetches the main model output from the stored Parquet database. 

        Returns:
            - str: The main model output.
        '''
        
        if self.model_output_str is not None:
            return self.model_output_str
        else:
            fetched_output = self.mo_parquet.fetch_model_output(self.bpv_idx)
            return fetched_output
    
    def get_bpv_idx(self):
        '''
        Returns the base prompt variation index.

        Returns:
            - tuple of ints: The base prompt variation index.
        '''
        return self.bpv_idx
    
    def write_output(self):
        '''
        Writes the main model output to the stored Parquet database.
        '''
        if self.model_output_str is None:
            raise ValueError("Cannot save an empty model output.")
        self.mo_parquet.insert_model_outputs([(self.bpv_idx, self.model_output_str)])

    def get_bpv_idx_and_model_output(self):
        '''
        Returns the bpv_idx and model_output_str as a tuple.
        '''
        full_str = self.get_output()
        return (self.bpv_idx, full_str)
    
    def read_output(self):
        '''
        Reads the main model output from the stored Parquet database.
        '''
        self.model_output_str = self.mo_parquet.fetch_model_output(self.bpv_idx)
