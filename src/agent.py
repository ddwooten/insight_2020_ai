# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import numpy as np
import tensorflow as tf
from sklearn import metrics as sk
import pdb

class agent:

    def __init__(self):
        
        self.accuracy_value = 0.0 
         
        self.factors = None

        self.loss = None

        self.loss_value = 0.0

        self.model = None

        self.model_name = None

        self.name_list = None

        self.num_correct = 0

        self.num_wrong = 0

        self.optimizer = None

        self.pred = None

        self.targets = None

        self.is_train = None
    
    def add_model(self):
        """This function calls the appropriate model builder"""
        
        if self.model_name == 'lstm':

            self.add_model_lstm()

    def add_model_lstm(self):
        """This function buils a basic lstm model"""

        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.LSTM(20, input_shape=(300,len(self.name_list) + 15)))

        self.model.add(tf.keras.layers.Dense(len(self.name_list), 
                                             activation = 'softmax'))

        # Don't forget an optimizer!

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    def custom_accuracy(self):
        """This function recreates the tf categorical accuracy 
        function. This is required as a workaround to bug #11749 for keras."""
        
        # If the maximum value (whos index is given by np.argmax) corresponds
        # with the correct value for self.targets value will have a value of 1,
        # otherwise, 0

        value = self.targets[np.argmax(self.pred)]

        if value > 0:

            self.num_correct += 1

        else:

            self.num_wrong += 1


        self.accuracy_value = (float(self.num_correct) / float((self.num_correct + self.num_wrong))) * 100.0
        
    def custom_loss(self):
        """This function manages the custom loss for the RL agent"""

        self.pred = self.model(self.factors, training = self.is_train)

        tensor_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        return(tensor_loss(self.targets, self.pred))

    def factorize(self, user_history):
        """This function converts a given user history to a factorized array
        for input into a model by calling the appropriate factorization function
        for that model class"""

        if self.model_name == 'lstm':

            self.factorize_lstm(user_history)

    def factorize_lstm(self, user_history):
        """This function factorizes a given user history, or batch of user
        histories, into factors for an lstm model"""

        # Reset the holding arrays

        self.factors = np.zeros((1, 300, 
                        (15 + len(self.name_list))))

        # This i here is to conform with tensorflow input expectations

        i = 0

        j = 0

        for index, row in user_history.iterrows():

            # The last entry in a history is the one we attempt to predict

            if j == (user_history.shape[0] - 1):

                self.targets = self.get_targets(row)

                break
            
            # Truncating maximum history to ~1 day of continuous listening

            if j == 300:

                self.targets = self.get_targets(row)

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

            # Using the Pandas one hot feature causes a memory explosion
            # So here we dynamically create the one hot vectors

            for k in range(15, len(self.name_list)):

                if self.name_list[k] == row['track_id']:

                    self.factors[i, j, k] = 1.0

                k += 1
                
            j += 1

    def get_gradient(self):
        """This function computes and returns the gradients of the given 
        model"""

        with tf.GradientTape() as tape:

            loss_value = self.custom_loss()

        return (loss_value, tape.gradient(loss_value, 
                                self.model.trainable_variables))

    def get_name_list(self, data):
        """This function converts item hashes to unique integer tags with a
        corresponding decrpytion dictionary"""

        self.name_list = [''] * len(data.track_id.unique())
        
        i = 0

        for track in data.track_id.unique():

            self.name_list[i] = track

            i += 1

    def get_targets(self, row):
        """This function creates a one hot vector of the target track id"""

        output = [0] * len(self.name_list)

        i = 0

        for track in self.name_list:
            
            if track == row['track_id']:

                output[i] = 1

            i += 1

        return(output)

    def train(self, user_history):
        """This function manages the training of the model based on the provided
        data"""

        self.factorize(user_history)

        loss_value, gradients = self.get_gradient() 

        self.optimizer.apply_gradients(zip(gradients, 
                                        self.model.trainable_variables))

        self.loss_value = self.loss(loss_value)

        self.custom_accuracy()

    def querry(self, user_history):
        """This function, given a user history, attempts to provide a suitable
        recommendation"""

        self.factorize(user_history)

        self.pred = self.model(self.factors, training = self.is_train)

        print("The recommendation is {}.\n".format(self.name_list[np.argmax(self.pred)]))

    def ready_agent(self, data, model_path, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.get_name_list(data)

        self.model_name = model_path 

        self.is_train = train 

        self.model = tf.saved_model.load(model_path)

        if self.model is not None:
            
            print("Model {} sucessuflly loaded.\n".format(model_path))

    def wake_agent(self, data, name, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.get_name_list(data)

        self.model_name = name

        self.is_train = train 

        self.loss = tf.keras.metrics.Mean()

        self.add_model()
