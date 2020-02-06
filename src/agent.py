# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import numpy as np
import tensorflow as tf
from sklearn import metrics as sk
import pdb

class agent:

    def __init__(self):
        
        self.accuracy_value_actor = 0.0 

        self.accuracy_value_critic = 0.0
         
        self.factors_actor = None

        self.factors_critic = None

        self.loss_value_actor = 0.0

        self.loss_value_critic = 0.0

        self.is_train = None

        self.model_actor = None

        self.model_critic = None

        self.model_name = None

        self.name_list = None

        self.num_rewards = 0

        self.optimizer = None

        self.pred = None

        self.targets = None

        self.total_reward = 0.0
    
    def add_model(self):
        """This function calls the appropriate model builder"""
        
        if self.model_name == 'lstm':

            self.add_model_lstm()

    def add_model_lstm(self):
        """This function buils a lstm model"""

        self.model_actor = tf.keras.Sequential()

        self.model_actor.add(tf.keras.layers.LSTM(20, input_shape=(300, 15)))

        self.model_actor.add(tf.keras.layers.Dense(11, 
                                             activation = 'relu'))

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

        self.pred = self.model_actor(self.factors, training = self.is_train)

        tensor_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        return(tensor_loss(self.targets, self.pred))

    def factorize(self, user_history):
        """This function converts a given user history to a factorized array
        for input into a model by calling the appropriate factorization function
        for that model class"""

        if self.model_actor_name == 'lstm':

            self.factorize_lstm(user_history)

    def factorize_lstm(self, user_history):
        """This function factorizes a given user history, or batch of user
        histories, into factors for an lstm model"""

        # Reset the holding arrays

        self.factors_agent = np.zeros((1, 300, 15)

        self.factors_critic = np.zeros((1, 300, 13)

        # This i here is to conform with tensorflow input expectations

        i = 0

        j = 0

        for index, row in user_history.iterrows():

            # The last entry in a history is the one we attempt to predict

            if j == (user_history.shape[0] - 1):

                break
            
            # Truncating maximum history to ~1 day of continuous listening

            if j == 300:

                break

            self.factors_agent[i, j, 0] = row['score']

            self.factors_critic[i, j, 0] = row['avg']

            self.factors_agent[i, j, 1] = row['instrumentalness']

            self.factors_citic[i, j, 1] = row['instrumentalness']

            self.factors_agent[i, j, 2] = row['liveness']

            self.factors_critic[i, j, 2] = row['liveness']

            self.factors_agent[i, j, 3] = row['speechiness']

            self.factors_critic[i, j, 3] = row['speechiness']

            self.factors_agent[i, j, 4] = row['danceability']

            self.factors_critic[i, j, 4] = row['danceability']

            self.factors_agent[i, j, 5] = row['valence']

            self.factors_critic[i, j, 5] = row['valence']

            self.factors_agent[i, j, 6] = row['loudness']

            self.factors_critic[i, j, 6] = row['loudness']

            self.factors_agent[i, j, 7] = row['tempo']

            self.factors_critic[i, j, 7] = row['tempo']

            self.factors_agent[i, j, 8] = row['acousticness']

            self.factors_critic[i, j, 8] = row['acousticness']

            self.factors_agent[i, j, 9] = row['energy']

            self.factors_critic[i, j, 9] = row['energy']

            self.factors_agent[i, j, 10] = row['mode']

            self.factors_critic[i, j, 10] = row['mode']

            self.factors_agent[i, j, 11] = row['key']

            self.factors_critic[i, j, 11] = row['key']

            self.factors_agent[i, j, 12] = row['day_w']

            self.factors_critic[i, j, 12] = row['sd']

            self.factors_agent[i, j, 13] = row['day_m']

            self.factors_agent[i, j, 14] = row['hour_d']

            j += 1

        i += 1

    def get_gradient(self):
        """This function computes and returns the gradients of the given 
        model"""

        with tf.GradientTape() as tape:

            loss_value = self.custom_loss()

        return (loss_value, tape.gradient(loss_value, 
                                self.model_actor.trainable_variables))

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

    def predict(self, user_history):
        """This function manages the training of the model based on the provided
        data"""

        self.factorize(user_history)

        self.pred = self.model_actor(self.factors, training = self.is_train)

    def propogate(self, divergence):
        """This function propogates the loss through the actor and critic"""

        loss_value, gradients = self.get_gradient() 

        self.optimizer.apply_gradients(zip(gradients, 
                                        self.model_actor.trainable_variables))

        self.loss_value = self.loss(loss_value)

        self.custom_accuracy()

    def querry(self, user_history):
        """This function, given a user history, attempts to provide a suitable
        recommendation"""

        self.factorize(user_history)

        self.pred = self.model_actor(self.factors, training = self.is_train)

        print("The recommendation is {}.\n".format(self.name_list[np.argmax(self.pred)]))

    def ready_agent(self, data, model_path, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.get_name_list(data)

        self.model_actor_name = model_path 

        self.is_train = train 

        self.model_actor = tf.saved_model.load(model_path)

        if self.model_actor is not None:
            
            print("Model {} sucessuflly loaded.\n".format(model_path))

    def wake_agent(self, data, name, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.get_name_list(data)

        self.model_name = name

        self.is_train = train 

        self.add_model()
