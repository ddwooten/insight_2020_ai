# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import numpy as np
import torch
import math
from sklearn import metrics as sk
import pdb
from model import AgentModel, CriticModel

class agent:

    def __init__(self):

        self.critic_loss = None
        
        self.factors_agent = None

        self.factors_critic = None

        self.history_len = 0

        self.is_train = None

        self.loss_agent = None

        self.loss_critic = None

        self.model_agent = None

        self.model_critic = None

        self.optimizer_agent = None

        self.optimizer_critic = None

        self.pred = None

        self.reward = None

    def add_model(self):
        """This function calls the appropriate model builder"""
        
        self.model_agent = AgentModel(15, 20, 11)

        self.model_critic = CriticModel(13, 21, 10, 0)

        self.set_model_weights(self.model_agent)

        self.set_model_weights(self.model_critic)

        self.optimizer_agent = torch.optim.Adam(self.model_agent.parameters(),
                                               lr = 0.001)

        self.optimizer_critic = torch.optim.Adam(self.model_critic.parameters(),
                                               lr = 0.001)

        self.loss_agent = torch.nn.MSELoss()

        self.loss_critic = torch.nn.MSELoss()

    def add_prediction(self, prediction):
        """This function concatenates the prediciton with the critic input"""
        
        i = 0

        j = self.history_len

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
        """This function factorizes a given user history, or batch of user
        histories, into factors for an lstm model"""

        # Reset the holding arrays

        self.factors_agent = np.zeros((1, 20, 15))

        self.factors_critic = np.zeros((1, 21, 13))

        # This i here is to conform with tensorflow input expectations

        i = 0

        j = 0

        for index, row in user_history.iterrows():

            # The last entry in a history is the one we attempt to predict

            if j == (user_history.shape[0] - 1):

                break
            
            # Truncating maximum history to ~1 day of continuous listening

            if j == 20:

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

    def get_agent_reward(self, repeat):
        """This function gets the agent reward""" 

        # if the track is something the user has heard before take the reward
        # the (1/2)

        if repeat > 0:

            reward =  math.pow(self.reward,0.5)

        else:

            reward = self.reward

        # Due to the square in the operation the magnitue of rward is limited
        # to 1E-7 due to machine precision concerns - verfied through testing

        if reward > 0.9999999:

            reward = 0.9999999 

        reward = torch.tensor([reward], requires_grad = True)

        self.reward = reward

    def get_critic_loss(self, current_user_history):
        """This function get the critic loss"""

        selection_array = current_user_history[['instrumentalness', 'liveness','speechiness', 'danceability', 'valence', 'loudness', 'tempo','acousticness', 'energy', 'm', 'k']]
       
        critic_loss = 0.0

        selection_array = selection_array.to_numpy()

        selection_array = selection_array[-10:]

        sel_len = selection_array.shape[0]

        distances = np.zeros((sel_len - 1) * (sel_len - 1))

        for i in range(sel_len - 1):
            
            for j in range(sel_len - 1):

                distances[j + j*i] = np.sqrt(np.sum(np.power((selection_array[j] - selection_array[i]),2)))

        total_possible = abs(current_user_history['sd'].values[0] - np.std(distances))

        distances = np.zeros((sel_len) * (sel_len))

        for i in range(sel_len):
            
            for j in range(sel_len):

                distances[j + j*i] = np.sqrt(np.sum(np.power((selection_array[j] - selection_array[i]),2)))
        
        achieved = abs(current_user_history['sd'].values[0] - np.std(distances))

        total_possible = torch.tensor([total_possible])

        achieved = torch.tensor([achieved], requires_grad = True)

        critic_loss = self.loss_critic(achieved,total_possible)

        critic_loss = torch.tensor([critic_loss], requires_grad = True)

        self.critic_loss = critic_loss

        return(critic_loss)

    def predict(self, user_history):
        """This function manages the training of the model based on the provided
        data"""

        self.factorize(user_history)

        self.pred = self.model_agent(torch.Tensor(self.factors_agent))

    def propogate(self, current_user_history, prediction, repeat):
        """This function propogates the loss through the actor and critic"""

        self.add_prediction(prediction)

        self.model_agent.zero_grad()

        self.model_critic.zero_grad()

        self.reward = self.model_critic(torch.Tensor(self.factors_critic))

        self.get_agent_reward(repeat)
        
        agent_loss = self.loss_agent(self.reward, torch.tensor([1.0]))
        
        self.optimizer_agent.step(agent_loss.backward())

        critic_loss = self.get_critic_loss(current_user_history)

        critic_loss = critic_loss

        self.optimizer_critic.step(critic_loss.backward())

    def ready_agent(self, agent_model_path, critic_model_path, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.is_train = train 

        self.model_agent = torch.load(agent_model_path)

        self.model_critic = torch.load(critic_model_path)

        if self.model_agent is not None:
            
            print("Actor Model {} sucessuflly loaded.\n".format(agent_model_path))

        self.model_critic = torch.load(critic_model_path)

        if self.model_agent is not None:
            
            print("Critic Model {} sucessuflly loaded.\n".format(critic_model_path))

    def set_model_weights(self, model):
        """This function initilizes the weights in pytorch model"""

        classname = model.__class__.__name__

        if classname.find('Linear') != -1:

            n = model.in_features

            y = 1.0 / np.sqrt(n)

            model.weight.data.uniform_(-y,y)

            model.bias.data.fill(0)

    def wake_agent(self, train):
        """This function sets up a working agent - one complete with a loss
        function and a model"""

        self.is_train = train 

        self.add_model()
