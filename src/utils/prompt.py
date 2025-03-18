'''
This class will be used to manage Prompts. It will include methods to create prompt, access prompts, write prompts, read prompts.
'''

from utils.data_handler import BasePromptDB
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

class ValidationScore:
    '''
    Manages the validation scores for prompt variations.
    This class provides methods to create, access, update, save, and load validation model scores.
    It also calculates aggregated scores across rubric sections.
    '''

    def __init__(self, bp_idx):
        '''
        Initialize a validation score with a base prompt index, for all prompt variations of that base prompt.

        Args:
            - bp_idx (tuple of ints): The base prompt index.
        '''
        self.bp_idx = bp_idx # Should be (n, -1)
        self.score = {}

    def store_scores(self, scores):
        """Stores the validation scores from the validator model inference."""
        self.score = scores
        # This is wrong I need to add something about parsing a tuple of 5 numbers.

    def get_scores(self):
        """Returns the validation scores."""
        return self.score
    
    def update_score(self, section, new_score):
        """Updates the validation score for a specific rubric section."""
        if section in self.score:
            self.score[section] = new_score
        else:
            raise ValueError(f"Section {section} not found in validation score.")
        
    def calculate_aggregated_score(self, method='mean'):
        """Calculates the aggregated score across rubric sections."""
        if not self.score:
            return None
        agg_scores = {}
        for section, scores in self.score.items():
            if method == "mean":
                agg_scores[section] = tuple(np.mean(scores, axis=0))
            elif method == "median":
                agg_scores[section] = tuple(np.median(scores, axis=0))
            else:
                raise ValueError("Unsupported aggregation method.")
        
        return agg_scores
    
    def get_validator_average_scores(self):
        """Returns the average scores assigned by the validator models."""
        if not self.score:
            return None
        # avg_scores = {}
        scores_matrix = list(self.score.values()) # List of (x, y, z) tuples
        avg_scores = tuple(np.mean(scores_matrix, axis=0)) # Average per validator
        
        return avg_scores
    
    def save_validation_scores(self, output_dir = "../data/validator_scores"):
        """Saves the validation scores to a Parquet file."""
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
      
        # Create DataFrame with bpv_idx column
        df = pd.DataFrame({"bp_idx": [self.bp_idx] * len(self.score) # Repeat bp_idx for all rubric sections
                           **self.score
        }) 

        # Ensure the filename is based on the base prompt index
        file_name = f"{output_dir}/validator_scores_{self.bp_idx[0]}.parquet"
        df.to_parquet(file_name, index=False)

class MainModelOutput:
    '''
    Class to be developed that will be used to manage the main model outputs. Will include methods to create main model output, access main model output, write main model output, read main model output, just like PromptVariation and BasePrompt. This class will be most similar to PromptVariation.
    '''