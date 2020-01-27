# This file contains the state class and all of its associated functions

import pandas as pd
import random

class state:

    def __init__(self, data):

        self.data = data 

        self.current_user_history = None

    def one_hot(self, data):
        """This function converts track ids to a one hot vector"""

        self.data =  pd.concat([self.data, pd.get_dummies(self.data.track_id)],
                                axis = 1)

        self.data = self.data.drop(axis=1, columns=['user_id','hashtag',
            'created_at', 'tweet_lang', 'current_track', 'previous_track',
            'zoned', 'aware'])

    def pull_random_user_history():
        """This function pull a random user history of random length"""

        user = random.choice(self.data.user_id.unique())

        history_start = random.randint(0, \
                        (self.data[self.data['user_id'] == user].shape[0] - 2))

        history_end = random.randint(history_start, \
                        (self.data[self.data['user_id'] == user].shape[0])

        self.current_user_history = self.data[self.data[\
            'user_id'] == user].iloc[history_start:history_end,]

    def report_record(self, loc):
        """This function prints the entry of data corresponding to the index
        given by loc"""

        print("{}\n".format(self.data.loc[loc]))

        

