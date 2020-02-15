# This file contains the agent class and all of its associated methods
# This 'agent' is a reinforcement learning agent.

import numpy as np
import torch
import math
import pdb
from model import AgentModel, CriticModel

class agent:

    def __init__(self):

        self.agent_loss = None

        self.critic_loss = None
        
        self.factors_agent = None

        self.factors_critic = None

        self.history_len = 0

        self.is_train = None

        self.loop = 0

        self.loss_agent = None

        self.loss_critic = None

        self.model_agent = None

        self.model_critic = None

        self.optimizer_agent = None

        self.optimizer_critic = None

        self.pred = None

        self.reward = None

        torch.set_default_dtype(torch.float64)

    def add_model(self):
        """This function calls the appropriate model builder"""
        
        self.model_agent = AgentModel(12, 6, 6)

        self.set_model_weights(self.model_agent)

        self.optimizer_agent = torch.optim.Adam(self.model_agent.parameters(),
                                               lr = 0.01)

        self.loss_agent = torch.nn.MSELoss()

    def add_prediction(self, prediction):
        """This function concatenates the prediciton with the critic input"""
        
        i = 0

        j = self.history_len

        self.factors[i, j, 0] = prediction['score']

        self.factors[i, j, 1] = prediction['r0']

        self.factors[i, j, 2] = prediction['r1']

        self.factors_critic[i, j, 3] = prediction['r2']

        self.factors_critic[i, j, 4] = prediction['r3']

        self.factors_critic[i, j, 5] = prediction['r4']

        self.factors_critic[i, j, 6] = prediction['r5']

        self.factors_critic[i, j, 7] = prediction['sd']

        self.factors_critic[i, j, 8] = prediction['avg']

        self.factors_critic[i, j, 9] = prediction['m']

        self.factors_critic[i, j, 10] = prediction['k']

    def custom_loss_critic(self, target, selection, selection_averages,
                           target_averages):
        """This returns the normalized cross correlation between target and
        selection"""

        # These lines here compute the cross-correlation between target and
        # selection

        top = np.multiply((selection - selection_averages), 
                          (target - target_averages))

        top_sum = np.sum(top, axis = 0)

        bottom_selection = np.power((selection - selection_averages),2)

        bottom_targets = np.power((target - target_averages), 2)

        bottom_selection_sum = np.sum(bottom_selection, axis = 0)

        bottom_targets_sum = np.sum(bottom_targets, axis = 0)

        bottom = np.sqrt(np.multiply(bottom_selection_sum,
                                     bottom_targets_sum))

        divided = np.divide(top_sum, bottom)

        divided = divided[~np.isnan(divided)]

        return(np.sum(divided))
            
    def factorize(self, user_history):
        """This function factorizes a given user history, or batch of user
        histories, into factors for an lstm model"""

        # Reset the holding arrays

        self.factors_agent = np.zeros((1, 4, 12))

        self.factors_critic = np.zeros((1, 21, 11))

        # This i here is to conform with tensorflow input expectations

        i = 0

        j = 0

        for index, row in user_history.iterrows():

            # The last entry in a history is the one we attempt to predict

            if j == (user_history.shape[0]):

                break

            if j == 4:

                break
            # In an act of data reduction and factor selection, I drop
            # all spotify embeddings and deploy my own
            
            self.factors_agent[i, j, 0] = row['score']

            self.factors_critic[i, j, 0] = row['score']

            self.factors_agent[i, j, 1] = row['r0']

            self.factors_critic[i, j, 1] = row['r0']

            self.factors_agent[i, j, 2] = row['r1']

            self.factors_critic[i, j, 2] = row['r1']

            self.factors_agent[i, j, 3] = row['r2']

            self.factors_critic[i, j, 3] = row['r2']

            self.factors_agent[i, j, 4] = row['r3']

            self.factors_critic[i, j, 4] = row['r3']

            self.factors_agent[i, j, 5] = row['r4']

            self.factors_critic[i, j, 5] = row['r4']

            self.factors_agent[i, j, 6] = row['r5']

            self.factors_critic[i, j, 6] = row['r5']

            self.factors_agent[i, j, 7] = row['m']

            self.factors_critic[i, j, 7] = row['m']

            self.factors_agent[i, j, 8] = row['k']

            self.factors_critic[i, j, 8] = row['k']

            self.factors_agent[i, j, 9] = row['day_w']

            self.factors_critic[i, j, 9] = row['sd']

            self.factors_agent[i, j, 10] = row['day_m']

            self.factors_critic[i, j, 10] = row['avg']

            self.factors_agent[i, j, 11] = row['hour_d']

            j += 1

        i += 1
        
        self.history_len = j

    def get_agent_reward(self, repeat):
        """This function gets the agent reward""" 

        # if the track is something the user has heard before take the reward
        # to the (1/2)

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

    def get_critic_loss(self, current_user_history, data):
        """This function get the critic loss"""

        user = data[data.user_id == current_user_history.user_id.values[0]]

        user = user[['r0','r1','r2','r3', 'r4', 'r5']]

        user_array = user.to_numpy()

        # In order to use handy dandy numpy list comprehensions, we need to
        # make an overly bulky array for the averages both for target and for
        # selection ( as pssed to self.custom_loss_critic)

        selection_averages = []

        selection_averages.append(np.average(current_user_history.r0.values))

        selection_averages.append(np.average(current_user_history.r1.values))

        selection_averages.append(np.average(current_user_history.r2.values))

        selection_averages.append(np.average(current_user_history.r3.values))

        selection_averages.append(np.average(current_user_history.r4.values))

        selection_averages.append(np.average(current_user_history.r5.values))

        selection_averages = np.array(selection_averages)

        # This line here gives selection_averages a 2nd dimension to match time
        # while the repeat command coppies these average values through the time
        # axis

        selection_averages = np.repeat(selection_averages[None,:], 
                                       current_user_history.shape[0],
                                       axis = 0)

        selection_averages = selection_averages[-10:]

        selection_array=current_user_history[['r0','r1','r2','r3', 'r4', 'r5']]

        selection_array = selection_array[-10:]

        selection_array = selection_array.to_numpy()

        # Here we repeat this process for the whole user history as reflected
        # byuser

        target_averages = []

        target_averages.append(np.average(user.r0.values))

        target_averages.append(np.average(user.r1.values))
       
        target_averages.append(np.average(user.r2.values))
       
        target_averages.append(np.average(user.r3.values))
       
        target_averages.append(np.average(user.r4.values))
       
        target_averages.append(np.average(user.r5.values))
        
        target_averages = np.array(target_averages)

        target_averages = np.repeat(target_averages[None, :],
                                    selection_array.shape[0],
                                    axis = 0)
        
        critic_loss = []

        end  = selection_array.shape[0]

        start = 0

        while end < user_array.shape[0]:
            
            critic_loss.append(self.custom_loss_critic(user_array[start:end,],
                                                selection_array,
                                                selection_averages,
                                                target_averages))

            start += 1

            end += 1

        if len(critic_loss) > 0:

            critic_loss = np.average(critic_loss)

        else:

            critic_loss = 0.0

        critic_loss = torch.tensor([critic_loss], requires_grad = True)

        self.critic_loss = critic_loss

    def predict(self, user_history):
        """This function manages the training of the model based on the provided
        data"""

        self.factorize(user_history)

        self.pred = self.model_agent(torch.Tensor(self.factors_agent[:,:-1,:]))

    def propagate(self):
        """This function propagates the loss through the actor and critic"""

        # Clear out the gradients from the last prediction

        self.loop += 1

        self.model_agent.zero_grad()

        # Get the critic reward

        agent_loss = self.loss_agent(self.pred,
                torch.tensor(self.factors_agent[0, self.history_len - 1, 1:7]))

        self.agent_loss = agent_loss.detach().numpy()

        self.optimizer_agent.step(agent_loss.backward())

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
        """This function initilizes the weights in a pytorch model"""

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
