# This file contains the state class and all of its associated functions

import pandas as pd
import random
import math
import pdb

class state:

    def __init__(self, data):

        self.current_user_history = None
        
        self.data = data 

        self.val_set = None

    def get_random_user_history(self):
        """This function pull a random user history of random length"""

        user = None

        while user is None:

            user = random.choice(self.data.user_id.unique())

            pdb.set_trace()

            if self.val_set is not None:

                if user in self.val_set:

                    user = None

            if self.data[self.data['user_id'] == user].shape[0] < 3:

                user = None

        history_start = 0

        history_end = 0

        while (history_end - history_start) < 3:

            history_start = random.randint(0, \
                        (self.data[self.data['user_id'] == user].shape[0] - 3))

            history_end = random.randint(history_start, \
                            (self.data[self.data['user_id'] == user].shape[0]))

        self.current_user_history = self.data[self.data[\
            'user_id'] == user].iloc[history_start:history_end,]

    def one_hot(self):
        """This function converts track ids to a one hot vector"""

        self.data =  pd.concat([self.data, pd.get_dummies(self.data.track_id)],
                                axis = 1)

        self.data = self.data.drop(axis=1, columns=['user_id','hashtag',
            'created_at', 'tweet_lang', 'current_track', 'previous_track',
            'zoned', 'aware'])

    def report_record(self, loc):
        """This function prints the entry of data corresponding to the index
        given by loc"""

        print("{}\n".format(self.data.loc[loc]))

        
    def set_aside_validation_set(self):
        """This function creates a list of user ids equal to 20% of the input
        data, these user ids are for validation"""

        sample_size = math.ceil(0.20 * self.data.shape[0])

        self.val_set = self.data.user_id.sample(sample_size)
