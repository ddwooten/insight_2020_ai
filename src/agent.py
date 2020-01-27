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

        self.name_dict = None
    
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

    def custom_loss(y_actual, y_pred):
        """This function manages the custom loss for the RL agent"""

    def factorize(self, user_history):
        """This function converts a given user history to a factorized array
        for input into a model by calling the appropriate factorization function
        for that model class"""

        if self.model_name == 'lstm':

            self.factorize_lstm(user_history)

    def factorize_lstm(self, user_history)
        """This function factorizes a given user history, or batch of user
        histories, into factors for an lstm model"""

        self.factors = np.zeroes(len(user_history), 300, 
                        user_history.shape[1])
        
        i = 0

        for history in user_history:

            j = 0

            for index, row in history.iterrows():

                if j == 300:

                    break

                k = 0

                for column in history.columns():

                    self.factors[i, j, k] = history[column][row]

                    k += 1

                j += 1

            i += 1

    def get_name_dict(self, data):
        """This function converts item hashes to unique integer tags with a
        corresponding decrpytion dictionary"""

        output = [{}, {}]

        for track in data.unique():

            output[0][track] = len(output[0])

            output[1][len(output[0])] = track

        self.name_dict = output

    def wake_agent(self, data, name):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.model_name = name

        self.name_dict = self.get_name_dict(data)

        self.add_model(self.model_name)


