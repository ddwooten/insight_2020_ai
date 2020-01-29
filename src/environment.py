# This file contains the evironment class and all of its methods
import pdb
class environment:

    def __init__(self, agent, state, setup_dict):

        self.agent = agent

        self.setup_dict = setup_dict

        self.state = state

    def results(self):
        """This function returns simulation results"""

        return(self.agent.accuracy.result(), self.agent.loss.result())

    def run_sim(self):
        """This function oversees execution of the simulation"""

        if self.setup_dict['train'] > 0:

            self.train_agent()

        else:

            self.querry_agent()
    
    def train_agent(self):
        """This function manages the training of an agent"""

        self.agent.wake_agent(self.state.data,
                                self.setup_dict['model'], 
                                self.setup_dict['train'])

        self.state.set_aside_validation_set()

        for epoch in range(self.setup_dict['epochs']):

            self.state.get_random_user_history()

            self.agent.train(self.state.current_user_history)

