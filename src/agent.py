# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import numpy as np
import tensorflow as tf
import math
from sklearn import metrics as sk
import pdb

class agent:

    def __init__(self):
        
        self.factors_agent = None

        self.factors_critic = None

        self.history_len = 0

        self.is_train = None

        self.model_agent = None

        self.model_critic = None

        self.model_name = None

        self.name_list = None

        self.optimizer = None

        self.pred = None

    def add_model(self):
        """This function calls the appropriate model builder"""
        
        if self.model_name == 'lstm':

            self.add_model_lstm()

    def add_model_lstm(self):
        """This function buils a lstm model"""
        
        # Create the agent model

        self.model_agent = tf.keras.Sequential()

        self.model_agent.add(tf.keras.layers.LSTM(300, input_shape=(300, 15)))

        self.model_agent.add(tf.keras.layers.Dense(11, 
                                             activation = 'softmax'))
        
        # Create the critic model

        self.model_critic = tf.keras.Sequential()

        self.model_critic.add(tf.keras.layers.LSTM(301, input_shape=(301, 13)))

        self.model_critic.add(tf.keras.layers.Dense(150, 
                                             activation = 'relu'))

        self.model_critic.add(tf.keras.layers.Dense(15, 
                                             activation = 'relu'))

        self.model_critic.add(tf.keras.layers.Dense(1, 
                                             activation = 'softmax'))

        # Don't forget an optimizer!

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
   
    def add_prediction(self, prediction):
        """This function concatenates the prediciton with the critic input"""
        
        i = 0

        j = self.history_len + 1

        self.factors_critic[i, j, 0] = prediction['avg']

        self.factors_critic[i, j, 1] = prediction['instrumentalness']

        self.factors_critic[i, j, 2] = prediction['liveness']

        self.factors_critic[i, j, 3] = prediction['speechiness']

        self.factors_critic[i, j, 4] = prediction['danceability']

        self.factors_critic[i, j, 5] = prediction['valence']

        self.factors_critic[i, j, 6] = prediction['loudness']

        self.factors_critic[i, j, 7] = prediction['tempo']

        self.factors_critic[i, j, 8] = prediction['acousticness']

        self.factors_critic[i, j, 9] = prediction['energy']

        self.factors_critic[i, j, 10] = prediction['m']

        self.factors_critic[i, j, 11] = prediction['k']

        self.factors_critic[i, j, 12] = prediction['sd']

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

        self.factors_agent = np.zeros((1, 300, 15))

        self.factors_critic = np.zeros((1, 301, 13))

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

            self.factors_critic[i, j, 1] = row['instrumentalness']

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

            self.factors_agent[i, j, 10] = row['m']

            self.factors_critic[i, j, 10] = row['m']

            self.factors_agent[i, j, 11] = row['k']

            self.factors_critic[i, j, 11] = row['k']

            self.factors_agent[i, j, 12] = row['day_w']

            self.factors_critic[i, j, 12] = row['sd']

            self.factors_agent[i, j, 13] = row['day_m']

            self.factors_agent[i, j, 14] = row['hour_d']

            j += 1

        i += 1

        self.history_len = j

    def get_gradient(self, divergence, prediction):
        """This function computes and returns the gradients of the given 
        model"""

        # if the track is something the user has heard before take the reward
        # the (1/2)

        if prediction.rating.values[0] > 0:

            agent_loss = 1.0 - math.pow(self.reward,0.5)

        else:

            agent_loss = 1.0 - self.reward

        with tf.GradientTape() as tape:

            agent_gradients = tape.gradient(agent_loss,
                                            self.model_agent.trainable_variables)

            critic_gradients = tape.gradient(divergence,
                                            self.model_critic.trainable_variables)

        return (agent_gradients, critic_gradients)

    def predict(self, user_history):
        """This function manages the training of the model based on the provided
        data"""

        self.factorize(user_history)

        self.pred = self.model_agent(self.factors_agent, training=self.is_train)

    def propogate(self, divergence, prediction):
        """This function propogates the loss through the actor and critic"""

        self.add_prediction(prediction)

        self.reward = self.model_critic(self.factors_critic, training=self.is_train)
        
        gradients_agent, gradients_critic = self.get_gradient(divergence, 
                                                              prediction) 

        self.optimizer.apply_gradients(zip(gradients_agent, 
                                        self.model_agent.trainable_variables))

        self.optimizer.apply_gradients(zip(gradients_critic, 
                                        self.model_agent.trainable_variables))

    def ready_agent(self, data, agent_model_path, critic_model_path, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.model_agent_name = model_path 

        self.is_train = train 

        self.model_agent = tf.saved_model.load(agent_model_path)

        if self.model_agent is not None:
            
            print("Actor Model {} sucessuflly loaded.\n".format(agent_model_path))

        self.model_critic = tf.saved_model.load(critic_model_path)

        if self.model_agent is not None:
            
            print("Critic Model {} sucessuflly loaded.\n".format(critic_model_path))

    def wake_agent(self, name, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.model_name = name

        self.is_train = train 

        self.add_model()
