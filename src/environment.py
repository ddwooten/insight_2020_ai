# This file contains the evironment class and all of its methods

class environment:

    def __init__(self, agent, state, setup_dict):

        self.agent = agent

        self.state = state

        self.setup_dict = setup_dict

    def run_sim(self):
        """This function oversees execution of the simulation"""

        self.state.one_hot()

        self.agent.wake_agent(self.setup_dict['model'])

        if self.setup_dict['train'] == 'yes':

            self.train_agent()

        else:

            self.querry_agent()
    
    def train_agent(self):
        """This function manages the training of an agent"""

        self.agent.wake_agent(self.state.data.track_id)


