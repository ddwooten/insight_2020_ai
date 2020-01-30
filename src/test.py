#!/home/dwooten/anaconda3/bin/python3

# test.py

# This file contains and runs all tests for the host code base

import re as re 
import datetime
from utilities import *

def read_setup_test():
    """Tests the setup reader"""

    test_value = 0

    eval_case = {'train':1, 'data':'../data/clean.pk', 'model':'lstm','epochs':1000, 'model_path':'../models/in_service.hdf5'}

    try:

        setup = open("setup.txt","r")

    except OSError:

        test_value = 0
        
        return(test_value)

    test_case = read_setup(setup)

    setup.close()

    if eval_case == test_case:

        test_value = 1

    return(test_value)

def test():
    """Executes unit, integration, and system tests. Outputs results to
    test.test."""

    print("Begining RL for recommender systems testing program.\n")

    print("Created on {}.\n\n".format(datetime.datetime.now()))

    tests = {}

    try:

        out_file = open("test.test","w")

    except OSError:

        print("Failed to open test.test. Exiting gracefully.\n\n")

        return()

    tests['read_setup_test'] = read_setup_test()

    total_tests = 0

    pass_tests = 0

    fail_tests = 0

    for key in tests.keys():

        total_tests += 1

        if tests[key] > 0:

            pass_tests += 1

            out_file.write("Test: {}: PASS\n\n".format(key))

        else:

            fail_tests += 1

            out_file.write("Test: {}: FAIL\n\n".format(key))

    out_file.write("Tests passed/total: {} / {}\n\n".format(pass_tests,
                                                           total_tests))

    out_file.close()

    print("Ending testing program.\n\n")

if __name__ == '__main__':

    test()
