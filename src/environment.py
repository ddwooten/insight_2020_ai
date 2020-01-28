# This file contains the evironment class and all of its methods

class environment:

    def __init__(self, agent, state, setup_dict):

        self.agent = agent

        self.setup_dict = setup_dict

        self.state = state

    def results(self):
        """This function returns simulation results"""

        return(0)

    def run_sim(self):
        """This function oversees execution of the simulation"""

        self.agent.wake_agent(self.state.data, self.setup_dict['model'])

        if self.setup_dict['train'] > 0:

            self.train_agent()

        else:

            self.querry_agent()
    
    def train_agent(self):
        """This function manages the training of an agent"""

        self.agent.wake_agent(self.setup['model'], self.setup['train'])

        self.state.set_aside_validation_set()

        for epoch in range(self.setup['epochs']):

            self.agent.train(self.state.get_random_user_history())

