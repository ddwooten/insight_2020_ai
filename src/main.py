#!/home/dwooten/anaconda3/bin/python3

# main.py

import pandas as pd
import numpy as np
import tensorflow as tf
import re as re 
from agent import agent
from environment import environment
from state import state
from utilities import *
import pdb

def main():
    """Executes training and infernce of/from model"""

    print("\nBegining RL for recommender systems emulator.\n\n")

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

    if setup_dict['train'] > 0:

        env.run_sim()

        results = env.results()

        print("Accuracy:{}, Loss:{}".format(results[0], results[1]))

        print("Exiting program gracefully.\n\n")

    else:

        env.querry_agent()

        print("Exiting program gracefully.\n\n")
    
if __name__ == '__main__':

    main()
