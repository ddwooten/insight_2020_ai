# This file contains the evironment class and all of its methods

import pdb
import torch 
import math
import numpy as np
from datetime import datetime

class environment:

    def __init__(self, agent, state, setup_dict):

        self.agent = agent

        self.loss_agent = [] 

        self.loss_critic = []

        self.predictions = []

        self.setup_dict = setup_dict

        self.state = state

    def results(self):
        """This function returns simulation results"""

        return(self.agent.accuracy_value, self.agent.loss_value)

    def run_sim(self):
        """This function oversees execution of the simulation"""

        if self.setup_dict['train'] > 0:

            self.train_agent()

        else:

            report = self.querry_agent()

            print("Rec: {}.\n".format(report))
    
    def train_agent(self):
        """This function manages the training of an agent"""

        # Reset the loss variables

        self.agent_loss = []

        self.critic_loss = []

        self.agent.wake_agent(self.setup_dict['train'])

        for epoch in range(self.setup_dict['epochs']):

            self.state.get_random_user_history()

            self.agent.predict(self.state.current_user_history)

            self.state.produce(self.agent.pred, 0.01)

            self.agent.propogate(self.state.current_user_history, 
                                 self.state.product,
                                 self.state.repeat)

            self.loss_agent.append(math.pow((1.0 - self.agent.reward),2))

            self.loss_critic.append(self.agent.critic_loss)

            print("Actor Loss: {}\nCritic Loss:{}\n\n".format((1.0 - self.agent.reward.item()), self.agent.critic_loss.item()))

            if np.mod(100, epoch) == 0: 

                # Save the model now that it has been trained

                model_name = 'agent_model_{}'.format(datetime.now())

                torch.save(self.agent.model_agent, '../models/' + model_name)

                model_name = 'critic_model_{}'.format(datetime.now())

                torch.save(self.agent.model_critic, '../models/' + model_name)
        
        model_name = 'end_agent_model_{}'.format(datetime.now())

        torch.save(self.agent.model_agent, '../models/' + model_name)

        model_name = 'end_critic_model_{}'.format(datetime.now())

        torch.save(self.agent.model_critic, '../models/' + model_name)

        # Save the loss arrays as csvs

        np.savetxt('./Agent_loss.csv', self.agent_loss, delimiter=',')

        np.savetxt('./Critic_loss.csv', self.critic_loss, delimiter=',')

    def querry_agent(self):
        """This function requests a recommendation of an agent"""

        self.agent.ready_agent(self.setup_dict['model_agent_path'], 
                               self.setup_dict['model_critic_path'], 
                               self.setup_dict['train'])

        self.state.get_sample_input()

        self.agent.predict(self.state.current_user_history)

        self.state.produce(self.agent.pred, 0.1)

        self.predictions.append(self.state.product.track_id.values[0])

        return(self.state.product.track_id.values[0])

