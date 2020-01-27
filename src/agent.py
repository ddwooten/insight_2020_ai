# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import tensorflow as tf

class agent:

    def __init__(self):
        
        self.model = None

        self.model_name = None

        self.loss = None

        self.train = None

        self.factors = None

        self.name_list = None
    
    def add_model(self):
        """This function calls the appropriate model builder"""

        if self.model_name == 'lstm':

            self.model = self.add_model_lstm()

    def add_model_lstm(self):
        """This function buils a basic lstm model"""

        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.LSTM(300))

        self.model.add(tf.keras.layers.Dense(len(self.name_dict[0]), 
                                             activation = 'softmax'))

    def custom_loss(y_actual):
        """This function manages the custom loss for the RL agent"""

        y_pred = self.model(self.factors, training = self.train)
        
        tensor_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                                        from_logits=False)

        return(tensor_loss(y_actual, y_pred)

    def factorize(self, user_history):
        """This function converts a given user history to a factorized array
        for input into a model by calling the appropriate factorization function
        for that model class"""

        if self.model_name == 'lstm':

            self.factorize_lstm(user_history)

    def factorize_lstm(self, user_history):
        """This function factorizes a given user history, or batch of user
        histories, into factors for an lstm model"""

        self.factors = np.zeroes(len(user_history), 300, 
                        15 + len(self.name_list))
        
        i = 0

        for history in user_history:

            j = 0

            for index, row in history.iterrows():

                if j == 300:

                    break

		self.factors[i, j, 0] = row['score']

                self.factors[i, j, 1] = row['instrumentalness']

                self.factors[i, j, 2] = row['liveness']

                self.factors[i, j, 3] = row['speechiness']

                self.factors[i, j, 4] = row['danceability']

                self.factors[i, j, 5] = row['valence']

                self.factors[i, j, 6] = row['loudness']

                self.factors[i, j, 7] = row['tempo']

                self.factors[i, j, 8] = row['acousticness']

                self.factors[i, j, 9] = row['energy']

                self.factors[i, j, 10] = row['mode']

                self.factors[i, j, 11] = row['key']

                self.factors[i, j, 12] = row['day_w']

                self.factors[i, j, 13] = row['day_m']

                self.factors[i, j, 14] = row['hour_d']

                for k in range(15, len(self.name_list)):

                    if self.name_list[k] == row['track_id']:

                        self.factors[i, j, k] = 1.0

                    k += 1
                    
                j += 1

            i += 1

    def get_name_list(self, data):
        """This function converts item hashes to unique integer tags with a
        corresponding decrpytion dictionary"""

        output = []
        
        i = 0

        for track in data.unique():

            output[i] = track

        self.name_list = output

    def wake_agent(self, data, name, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.model_name = name

        self.train = 1 if train == 'yes' else self.train = 0

        self.name_dict = self.get_name_dict(data)

        self.add_model(self.model_name)


