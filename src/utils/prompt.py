'''
This class will be used to manage Prompts. It will include methods to create prompt, access prompts, write prompts, read prompts.
'''

#from utils.data_handler import BasePromptDB, ValidationScoreParquet
#from configs.data_size_configs import NUM_RUBRIC_SECTIONS, SECTION_WEIGHTS
from configs.data_size_configs import *
import numpy as np
import os
from configs.root_paths import *
from utils.data_handler import * 
import pandas as pd


class BasePrompt:
    '''
    This class will be used to manage Base Prompts.
    It will include methods to create base prompt, access base prompt, write base prompt, read base prompt. Every component of the project that needs to access base prompts will use this class exclusively.
    '''

    def __init__(self, bp_idx, bp_db, prompt_str = None):
        '''
        Initialize a base prompt with a base prompt index. Whether to create a new base prompt, or access an existing base prompt, this base prompt class will be used.

        Args:
            - bp_idx (int): The base prompt index.
        '''
        self.bp_idx = bp_idx
        self.prompt_str = prompt_str
        self.bp_db = bp_db

    def get_base_prompt(self):
        '''
        Fetches the prompt from the database.
        If a database connection is not provided, a new connection is created and closed. If it is provided, it is used and not closed.

        Args:
            - db (BasePromptDB, optional): An existing database connection object.

        Returns:
            - str: The prompt string.
        '''

        # this if statement would never AND BELLA MEANS NEVER get triggered
        # if self.bp_db is None:

        #     self.bp_db = BasePromptDB()
        #     self.prompt = db.fetch_prompt(self.bpv_idx)
        #     db.close_connection()
        # else:
        #     self.prompt = db.fetch_prompt(self.bpv_idx)

        # return self.prompt

        if self.prompt_str is not None:
            return self.prompt_str
        else:
            return self.bp_db.fetch_prompt(self.bp_idx)

    def get_prompt_index(self):
        '''
        Returns the base prompt variation index.

        Returns:
            - tuple of ints: The base prompt variation index.
        '''
        return self.bp_idx

    # def save_base_prompt(self):
    #     This function is not be used right now.
    #     '''
    #     Saves the base prompt to the database.
    #     If a database connection is not provided, a new connection is created and closed. If it is provided, it is used and not closed.

    #     Args:
    #         - db (BasePromptDB, optional): An existing database connection object.
    #     '''
    #     if db is None:
    #         db = BasePromptDB()
    #         db.insert_base_prompts([(self.bpv_idx, self.prompt)])
    #         db.close_connection()
    #     else:
    #         db.insert_base_prompts([(self.bpv_idx, self.prompt)])


class PromptVariation:
    '''
    Class to be developed that will be used to manage Prompt Variations in a similar way BasePrompt manages base prompts. Will be initialized with a bpv_idx and will have methods to access, write, read prompt variations, just like above.
    We assume a a PromptVariationParquet object will be instantiated beforehand.
    '''
    def __init__(self, bpv_idx, pv_parquet, variation_str=None):
        '''
        Initialize a prompt variation with a base prompt variation index.

        Args:
            - bpv_idx (tuple of ints): The base prompt variation index.
        '''
        self.bpv_idx = bpv_idx
        self.variation_str = variation_str
        self.pv_parquet = pv_parquet
        # Add optional parameter that defaults to none called full_string

    def get_variation(self):
        '''
        Fetches the variation string from the corresponding prompt variation parquet file.

        Returns:
            - str: The prompt variation string.
        '''
        if self.variation_str is not None:
            return self.variation_str
        else:
            return self.pv_parquet.fetch_prompt_variation(self.bpv_idx)

    
    def get_bpv_idx(self):
        '''
        Returns the variation index from the corresponding prompt variation parquet file.

        Returns:
            - tuple of ints: The base prompt variation index.
        '''
        return self.bpv_idx
    
    def write_variation(self):
        '''
        Insert a single new variation string to the corresponding prompt variation parquet file.
        '''
        if self.variation_str is None:
            raise ValueError("Cannot insert an empty variation.")
        self.pv_parquet.insert_prompt_variations([(self.bpv_idx, self.variation_str)])

    def fetch_base_prompt(self):
        '''
        Fetches the base prompt string associated with this prompt variation.

        Returns:
            - str: The base prompt string if found, else None.
        '''
        return self.pv_parquet.fetch_base_prompt_str(self.bpv_idx)
    

    def get_base_prompt_and_variation(self):
        '''
        Fetches both the base prompt and the prompt variation string for this variation.

        Returns:
            - tuple: (base_prompt_string, prompt_variation_string)
        '''
        return self.pv_parquet.fetch_base_prompt_and_prompt_variation(self.bpv_idx)
    
    def read_variation(self):
        '''
        Reads the prompt variation from the stored Parquet database.
        '''
        self.variation_str = self.pv_parquet.fetch_prompt_variation(self.bpv_idx)


class ValidationScore: # instantiated for each bpv_idx

    '''
    Manages the validation scores for prompt variations.
    This class provides methods to create, access, update, save, and load validation model scores.
    It also calculates aggregated scores across rubric sections.
    '''
    def __init__(self, vs_parquet: ValidationScoreParquet, full_string = None): # ADD THAT string is optional
        '''
        Initialize a validation score with a base prompt index, for all prompt variations of that base prompt.

        Args:
            - bp_idx (tuple of ints): The base prompt index.
        '''
        self.bpv_idx # Use this to derive the bp_idx for file name
        self.full_string = full_string
        self.vs_parquet = vs_parquet
        self.scores = {f'section_{i+1}': [] for i in range(NUM_RUBRIC_SECTIONS)} # Initialize scores for each rubric section

    def parse_and_store_scores(self, string):
        """Parses the model scores from a string and stores them.
        Validator model output is a string of int scores in a tuple."""
        try:
            scores = list(map(int, string.split(',')))
            if len(scores != NUM_RUBRIC_SECTIONS):
                raise ValueError(f'The input string must contain exactly {NUM_RUBRIC_SECTIONS} scores.')
            for i in range(NUM_RUBRIC_SECTIONS):
                self.scores[f'section{i+1}'].append(scores[i])
        except ValueError as e:
            print(f'Error parsing the input string: {e}')

    def get_scores(self):
        """Returns the validation scores."""
        return self.scores

    def update_score(self, section, new_score):
        """Updates the validation score for a specific rubric section."""
        section_key = f'section_{section}'
        if section_key in self.scores:
            self.scores[section_key] = new_score
        else:
            raise ValueError(f"Section {section} not found in validation score.")

    # def get_validator_average_score(self):
    #     """Returns the average scores assigned by the validator models."""
    #     if not self.score:
    #         return None
    #     # avg_scores = {}
    #     scores_matrix = list(self.score.values()) # List of (x, y, z) tuples
    #     avg_scores = tuple(np.mean(scores_matrix, axis=0)) # Average per validator
    #     return avg_scores

    def calculate_average_section_scores(self):
        """Calculates the average score for each rubric section across all validator models."""
        if not self.scores:
            return None
        average_scores = {}
        for section in range(1, NUM_RUBRIC_SECTIONS + 1):
            section_key = f'section_{section}'
            if section_key in self.scores:
                average_scores[section_key] = np.mean(self.scores[section_key])
            else:
                raise ValueError(f"Section {section} not found in validation score.")
        return average_scores

    def calculate_total_score(self):
        """Calculates the total score across all rubric sections, considering weights from data_size_configs.RUBRIC_WEIGHTS."""
        if len(SECTION_WEIGHTS) != NUM_RUBRIC_SECTIONS:
            raise ValueError('The number of rubric sections does not match the number of weights.')
        if not np.isclose(sum(SECTION_WEIGHTS.values()), 1):
            raise ValueError('The rubric weights in SECTION_WEIGHTS must sum to 1.')
        
        average_scores = self.calculate_average_section_scores()
        if average_scores is None:
            return None
        total_score = 0
        for section, weight in SECTION_WEIGHTS.items():
            if section in average_scores:
                total_score += average_scores[section] * weight
            else:
                raise ValueError(f'Section {section} not found in average scores.')
        return total_score

    def save_validation_scores(self):
        """Returns the validation scores for passing to write_parquet_file()"""
        average_scores = self.calculate_average_section_scores()
        total_score = self.calculate_total_score()
        if average_scores is None or total_score is None:
            return None

        # Collect individual section scores as tuples
        individual_scores = [tuple(self.scores[f'section_{i+1}']) for i in range(NUM_RUBRIC_SECTIONS)]
        # Collect average section scores
        avg_scores_list = [average_scores[f'section_{i+1}_avg'] for i in range(NUM_RUBRIC_SECTIONS)]
        # Combine all scores into a single list
        scores_list = individual_scores + avg_scores_list + [total_score]
        return [self.bpv_idx, scores_list]
    
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
