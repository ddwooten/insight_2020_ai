# This file contains the state class and all of its associated functions

import pandas as pd
import random
import math
import pdb

class state:

    def __init__(self, data):

        self.current_user_history = None
        
        self.data = data 

        self.product = None

        self.val_set = None

    def divergence():
        """This function computes the minimum kl divergence between a given
        sequence and the total user history"""

        user = self.data[self.data.user_id==self.current_user_history.user_id]

        # Key is neglected as it is categorical not an actual scale or measure

        user_array = user.loc[['instrumentalness', 'liveness', 'speechiness',
                                'danceability', 'valence', 'loudness', 'tempo',
                                'acousticness', 'energy', 'm', 'k']]

        selection_array = self.current_user_history.append(self.produce).loc[['instrumentalness', 
                                'liveness','speechiness',
                                'danceability', 'valence', 'loudness', 'tempo',
                                'acousticness', 'energy', 'm', 'k']]
        
        user_array = user_array.to_numpy()

        selection_array = selection_array.to_numpy()

        start = 0

        end = self.current_user_history.shape[0] - 1

        self.divergence = 1E18

        while end < user.shape[0]:

            divergence = -np.sum(np.multiply(user_array[start:end,:], np.log(np.divide(selection_array, user_array[start:end,:]))))

            if divergence < self.divergence:

                self.divergence = divergence

            start = start + self.current_user_history.shape[0]

            end = end + self.current_user_history.shape[0]

    def get_random_user_history(self):
        """This function pull a random user history of random length"""

        user = None

        while user is None:

            user = random.choice(self.data.user_id.unique())

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

    def get_sample_input(self):
        """This function pull a random user history of random length"""

        user = None

        while user is None:

            user = random.choice(self.data.user_id.unique())

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

        self.data[self.data[\
            'user_id'] == user].iloc[history_start:history_end,].to_pickle('../data/predict.pk')

        self.current_user_history = pd.read_pickle('../data/predict.pk')

    def produce(self, attr, relax):
        """This function selects a song from data based on its match to spotify
        dimentions"""

        index = random.randint(0, 11)

        attr_low = [0] * 11

        attr_high = [0] * 11

        for i in range(11):

            if i == index:

                # attr is a two dimensional array (1,11) given by TF

                attr_low[i] = attr[0][i] - (relax/2.0)

                attr_high[i] = attr[0][i] + (relax/2.0)

            else:

                attr_low[i] = attr[0][i]

                attr_high[i] = attr[0][i]

        self.product = self.data[(self.data.instrumentalness >= attr_low[0]) &\
                                 (self.data.instrumentalness <= attr_high[0]) &\
                                 (self.data.liveness >= attr_low[1]) &\
                                 (self.data.liveness <= attr_high[1]) &\
                                 (self.data.speechiness >= attr_low[2]) &\
                                 (self.data.speechiness <= attr_high[2]) &\
                                 (self.data.danceability >= attr_low[3]) &\
                                 (self.data.danceability <= attr_high[3]) &\
                                 (self.data.valence >= attr_low[4]) &\
                                 (self.data.valence <= attr_high[4]) &\
                                 (self.data.loudness >= attr_low[5]) &\
                                 (self.data.loudness <= attr_high[5]) &\
                                 (self.data.tempo >= attr_low[6]) &\
                                 (self.data.tempo <= attr_high[6]) &\
                                 (self.data.acousticness >= attr_low[7]) &\
                                 (self.data.acousticness <= attr_high[7]) &\
                                 (self.data.energy >= attr_low[8]) &\
                                 (self.data.energy <= attr_high[8]) &\
                                 (self.data.m >= attr_low[9]) &\
                                 (self.data.m <= attr_high[9]) &\
                                 (self.data.k >= attr_low[10]) &\
                                 (self.data.k <= attr_high[10])].sample(1)

    def report_record(self, loc):
        """This function prints the entry of data corresponding to the index
        given by loc"""

        print("{}\n".format(self.data.loc[loc]))

        
    def set_aside_validation_set(self):
        """This function creates a list of user ids equal to 20% of the input
        data, these user ids are for validation"""

        sample_size = math.ceil(0.20 * self.data.shape[0])

        self.val_set = self.data.user_id.sample(sample_size)
