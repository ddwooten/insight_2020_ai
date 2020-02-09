# This file builds the model class as requested by PyTorch
# Credit to the PyTorch team and the excellent tutorial found at pytorch.org

import torch
import numpy as np
import pdb

class AgentModel(torch.nn.Module):

    def __init__(self, feature_len, lstm_len, dense_len):

        super(AgentModel, self).__init__()

        self.feature_len = feature_len

        self.lstm_len = lstm_len

        self.dense_len = dense_len

        self.lstm = torch.nn.LSTM(input_size = self.feature_len, 
                                  hidden_size = lstm_len,
                                  num_layers = 1,
                                  bias = True,
                                  batch_first = True,
                                  dropout = 0.0,
                                  bidirectional = False)

        self.dense = torch.nn.Linear(lstm_len, dense_len)

    def forward(self, batch):
        """ This function runs the prediciton sequence for the NN"""

        lstm_out, (last_hidden_state,last_cell_state) = self.lstm(batch)

        lstm_out = lstm_out.squeeze()[-1,:]

        lstm_out = torch.nn.functional.relu(lstm_out)

        dense_out = self.dense(lstm_out)

        embeddings_out = torch.nn.functional.logsigmoid(dense_out)

        return(embeddings_out)

class CriticModel(torch.nn.Module):

    def __init__(self, feature_len, lstm_len, dense_1_len, dense_2_len):

        super(CriticModel, self).__init__()

        self.feature_len = feature_len

        self.lstm_len = lstm_len

        self.dense_1_len = dense_1_len

        self.dense_2_len = dense_2_len

        self.lstm = torch.nn.LSTM(input_size = self.feature_len, 
                                  hidden_size = lstm_len,
                                  num_layers = 1,
                                  bias = True,
                                  batch_first = True,
                                  dropout = 0.0,
                                  bidirectional = False)

        self.dense_1 = torch.nn.Linear(lstm_len, dense_1_len)

        self.dense_2 = torch.nn.Linear(dense_1_len, dense_2_len)

        self.dense_3 = torch.nn.Linear(dense_2_len, 1)

    def forward(self, batch):
        """ This function runs the prediciton sequence for the NN"""

        lstm_out, (last_hidden_state,last_cell_state) = self.lstm(batch)

        lstm_out = lstm_out.squeeze()[-1,:]
        
        lstm_out = torch.nn.functional.relu(lstm_out)

        dense_1_out = self.dense_1(lstm_out)

        dense_1_out = torch.nn.functional.relu(dense_1_out)

        if self.dense_2_len > 0:

            dense_2_out = self.dense_2(dense_1_out)

            dense_2_out = torch.nn.functional.relu(dense_2_out)

        else:

            dense_2_out = dense_1_out

        dense_3_out = self.dense_3(dense_2_out)

        reward_out = torch.nn.functional.logsigmoid(dense_3_out)

        return(reward_out)
