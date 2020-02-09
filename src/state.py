# This file contains the state class and all of its associated functions

import pandas as pd
import numpy as np
import random
import math
import pdb

class state:

    def __init__(self, data):

        self.current_user_history = None
        
        self.data = data 

        self.product = None

        self.val_set = None

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

    def produce(self, attributes, relax):
        """This function selects a song from data based on its match to spotify
        dimentions"""

        index = random.randint(0, 11)

        # Slice off the tensor dimensions from torch, get the values

        attr = [0] * 11

        for i in range(11):

            attr[i] = attributes[0][0][i].item()

        attrlow = [0] * 11

        attrhigh = [0] * 11

        for i in range(11):

            if i == index:

                # attr is a two dimensional array (1,11) given by TF

                attrlow[i] = attr[i] * (1.0 - (relax/2.0))

                attrhigh[i] = attr[i] * (1.0 + (relax/2.0))

            else:

                attrlow[i] = attr[i]

                attrhigh[i] = attr[i]
        
        try:

            self.product = self.data[(self.data.instrumentalness >= attrlow[0]) &\
                                     (self.data.instrumentalness <= attrhigh[0]) &\
                                     (self.data.liveness >= attrlow[1]) &\
                                     (self.data.liveness <= attrhigh[1]) &\
                                     (self.data.speechiness >= attrlow[2]) &\
                                     (self.data.speechiness <= attrhigh[2]) &\
                                     (self.data.danceability >= attrlow[3]) &\
                                     (self.data.danceability <= attrhigh[3]) &\
                                     (self.data.valence >= attrlow[4]) &\
                                     (self.data.valence <= attrhigh[4]) &\
                                     (self.data.loudness >= attrlow[5]) &\
                                     (self.data.loudness <= attrhigh[5]) &\
                                     (self.data.tempo >= attrlow[6]) &\
                                     (self.data.tempo <= attrhigh[6]) &\
                                     (self.data.acousticness >= attrlow[7]) &\
                                     (self.data.acousticness <= attrhigh[7]) &\
                                     (self.data.energy >= attrlow[8]) &\
                                     (self.data.energy <= attrhigh[8]) &\
                                     (self.data.m >= attrlow[9]) &\
                                     (self.data.m <= attrhigh[9]) &\
                                     (self.data.k >= attrlow[10]) &\
                                     (self.data.k <= attrhigh[10])].sample(1)

        except ValueError:

            # If the attrs are so bad that no songs can be found, select a
            # random song
        
            self.product = self.data.sample(1)

        if self.product.track_id.values[0] in self.current_user_history.track_id.unique():
            self.repeat = 1

        else:

            self.repeat = 0

        self.current_user_history.append(self.product)

    def report_record(self, loc):
        """This function prints the entry of data corresponding to the index
        given by loc"""

        print("{}\n".format(self.data.loc[loc]))

        
    def set_aside_validation_set(self):
        """This function creates a list of user ids equal to 20% of the input
        data, these user ids are for validation"""

        sample_size = math.ceil(0.20 * self.data.shape[0])

        self.val_set = self.data.user_id.sample(sample_size)
