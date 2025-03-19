'''
This class will be used to manage Prompts. It will include methods to create prompt, access prompts, write prompts, read prompts.
'''

from utils.data_handler import BasePromptDB
from configs.data_size_configs import NUM_RUBRIC_SECTIONS, SECTION_WEIGHTS
import numpy as np
import pandas as pd
import os

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

class ValidationScore: # instantiated for each bpv_idx
    '''
    Manages the validation scores for prompt variations.
    This class provides methods to create, access, update, save, and load validation model scores.
    It also calculates aggregated scores across rubric sections.
    '''

    def __init__(self, bpv_idx):
        '''
        Initialize a validation score with a base prompt index, for all prompt variations of that base prompt.

        Args:
            - bp_idx (tuple of ints): The base prompt index.
        '''
        self.bpv_idx # Use this to derive the bp_idx for file name
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
    # CHANGE FUNCTION TO ALLOW FOR WEIGHTS
    # Add check that all weights sum to 1
    # Add check that size of RUBRIC_WEIGHTS is equal to NUM_RUBRIC_SECTIONS

    def save_validation_scores(self):
        """Returns the validation scores for passing to write_parquet_file()"""
        average_scores = self.calculate_average_section_scores()
        total_score = self.calculate_total_score()
        if average_scores is None or total_score is None:
            return None

        # Collect individual section scores
        individual_scorees = [self.scores[f'section_{i+1}'] for i in range(NUM_RUBRIC_SECTIONS)]
        # Flatten the list of individual scores
        flattened_individual_scores = [score for sublist in individual_scorees for score in sublist]
        # Collect average section scores
        avg_scores_list = [average_scores[f'section_{i+1}'] for i in range(NUM_RUBRIC_SECTIONS)]
        # Combine all scores into a single list
        scores_list = flattened_individual_scores + avg_scores_list + [total_score]
        return [self.bpv_idx, scores_list]

class MainModelOutput:
    '''
    Class to be developed that will be used to manage the main model outputs. Will include methods to create main model output, access main model output, write main model output, read main model output, just like PromptVariation and BasePrompt. This class will be most similar to PromptVariation.
    '''