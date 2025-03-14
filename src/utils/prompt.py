'''
This class will be used to manage Prompts. It will include methods to create prompt, access prompts, write prompts, read prompts.
'''

from utils.data_handler import BasePromptDB

class BasePrompt:
    '''
    This class will be used to manage Base Prompts.
    '''

    def __init__(self, bpv_idx):
        '''
        Initialize a base prompt with a base prompt variation index (given -1 for the second index of the tuple).

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