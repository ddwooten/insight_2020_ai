# This file builds the model class as requested by PyTorch
# Credit to the PyTorch team and the excellent tutorial found at pytorch.org

import torch
import numpy as np
import pdb

class AgentModel(torch.nn.Module):

    torch.set_default_dtype(torch.float64)

    def __init__(self):

        super(AgentModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1,3,2)

        self.conv21 = torch.nn.Conv2d(1,3,2)

        self.conv22 = torch.nn.Conv2d(1,3,2)

        self.conv23 = torch.nn.Conv2d(1,3,2)

        self.pool11 = torch.nn.MaxPool2d(2)

        self.pool12 = torch.nn.MaxPool2d(2)

        self.pool13 = torch.nn.MaxPool2d(2)

        self.dense_1 = torch.nn.Linear(15, 6)

    def forward(self, batch):
        """ This function runs the prediciton sequence for the NN"""
        
        conv1 = self.conv1(batch)

        conv1 = torch.tanh(conv1)

        pool11 = self.pool11(torch.index_select(conv1,1,torch.LongTensor([0])))

        pool12 = self.pool12(torch.index_select(conv1,1,torch.LongTensor([1])))

        pool13 = self.pool13(torch.index_select(conv1,1,torch.LongTensor([2])))

        prepped = torch.cat((pool11.flatten(), pool12.flatten(),
                            pool13.flatten()), 0)

        dense_1 = self.dense_1(prepped)

        return(dense_1)

class CriticModel(torch.nn.Module):

    torch.set_default_dtype(torch.float64)

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
        
        if self.dense_2_len > 0:

            self.dense_2 = torch.nn.Linear(dense_1_len, dense_2_len)

            self.dense_3 = torch.nn.Linear(dense_2_len, 1)

        else:

            self.dense_3 = torch.nn.Linear(dense_1_len, 1)

    def forward(self, batch):
        """ This function runs the prediciton sequence for the NN"""

        lstm_out, (last_hidden_state,last_cell_state) = self.lstm(batch)

        # This squeeze function pulls out the output signal for each lstm
        # cell for the last time step in the sequence

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

        reward_out = torch.sigmoid(dense_3_out)

        return(reward_out)
