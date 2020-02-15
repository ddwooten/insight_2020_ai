#!/home/dwooten/anaconda3/bin/python3

# main.py

import pandas as pd
import torch
from agent import agent
from environment import environment
from state import state
from utilities import *
import pdb

def main():
    """Executes training and infernce of/from model"""

    print("\nBegining RL for recommender systems emulator.\n\n")

    # Set a program wide default

    torch.set_default_dtype(torch.float64)

    try:

        setup = open("setup.txt","r")

    except OSError:

        print("Failed to open setup.txt. Exiting gracefully.\n\n")

        return()

    # Call read_setup to actually process the file-stream

    setup_dict = read_setup(setup)

    setup.close()

    print("Setup file sucessfully read.\n\n")

    # Retreive pandas pickle from location specified by setup file

    data = pd.read_pickle(setup_dict['data'])

    active_agent = agent() 

    active_state = state(data)

    env = environment(active_agent, active_state, setup_dict)

    env.run_sim()

    print("Exiting program gracefully.\n\n")

if __name__ == '__main__':

    main()
