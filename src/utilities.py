# A set of utility functions, stored here to not clutter main.py

import re as re

def read_setup(setup):
    """Reads the setup file stream for program configuation options as 
    expressed in the regex bellow. Outputs intake options as a dict."""

    output = {}

    # Default the model to inference 

    output['train'] = 0

    output['data'] = 'data.csv'

    train_pattern = re.compile(r'(?i)train,\s*([^,\n\r\t]*)')

    data_pattern = re.compile(r'(?i)data,\s*([^,\n\r\t]*)')

    model_pattern = re.compile(r'(?i)model,\s*([^,\n\r\t]*)')

    epochs_pattern = re.compile(r'(?i)epochs,\s*([0-9]+)')

    path_pattern = re.compile(r'(?i)model_path,\s*([^,\n\r\t]*)')

    yes_pattern = re.compile(r'(?i)yes')

    for line in setup:

        train_match = train_pattern.search(line)

        data_match = data_pattern.search(line)

        model_match = model_pattern.search(line)

        epochs_match = epochs_pattern.search(line)

        path_match = path_pattern.search(line)

        if train_match is not None:

            yes_match = yes_pattern.match(train_match.groups(1)[0])

            if yes_match is not None:

                output['train'] = 1

        if data_match is not None:

            output['data'] = data_match.groups(1)[0]

        if model_match is not None:

            output['model'] = model_match.groups(1)[0]

        if epochs_match is not None:

            output['epochs'] = int(epochs_match.groups(1)[0])

        if path_match is not None:

            output['model_path'] = path_match.groups(1)[0]


    return(output)
