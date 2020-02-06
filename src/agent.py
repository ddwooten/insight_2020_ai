# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import numpy as np
import tensorflow as tf
from sklearn import metrics as sk
import pdb

class agent:

    def __init__(self):
        
        self.factors_agent = None

        self.factors_critic = None

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
   
    def add_prediction(self, prediciton):
        """This function concatenates the prediciton with the critic input"""
        
        i = 0

        if self.current_user_history.shape[0] >= 300:

            j = 301

        else:

            j = self.current_user_history.shape[0]

        self.factors_critic[i, j, 0] = ['avg']

        self.factors_critic[i, j, 1] = ['instrumentalness']

        self.factors_critic[i, j, 2] = ['liveness']

        self.factors_critic[i, j, 3] = ['speechiness']

        self.factors_critic[i, j, 4] = ['danceability']

        self.factors_critic[i, j, 5] = ['valence']

        self.factors_critic[i, j, 6] = ['loudness']

        self.factors_critic[i, j, 7] = ['tempo']

        self.factors_critic[i, j, 8] = ['acousticness']

        self.factors_critic[i, j, 9] = ['energy']

        self.factors_critic[i, j, 10] = ['mode']

        self.factors_critic[i, j, 11] = ['k']

        self.factors_critic[i, j, 12] = ['sd']

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

            self.factors_agent[i, j, 10] = row['mode']

            self.factors_critic[i, j, 10] = row['mode']

            self.factors_agent[i, j, 11] = row['k']

            self.factors_critic[i, j, 11] = row['k']

            self.factors_agent[i, j, 12] = row['day_w']

            self.factors_critic[i, j, 12] = row['sd']

            self.factors_agent[i, j, 13] = row['day_m']

            self.factors_agent[i, j, 14] = row['hour_d']

            j += 1

        i += 1

    def get_gradient(self, divergence):
        """This function computes and returns the gradients of the given 
        model"""

        agent_gradients = tf.GradientTape.gradient((1.0 - self.reward),
                                        self.model_agent.trainable_variables)

        critic_gradients = tf.GradientTape.gradient(divergence,
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
        
        gradients_agent, gradients_critic = self.get_gradient(divergence) 

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
