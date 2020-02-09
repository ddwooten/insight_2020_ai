# This file builds the model class as requested by PyTorch
# Credit to the PyTorch team and the excellent tutorial found at pytorch.org

import torch

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

        lstm_out, _ = self.lstm(batch)

        dense_out = self.dense(lstm_out.view(self.lstm_len, -1))

        embeddings_out = torch.nn.functional.log_softmax(dense_out, dim=1)

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

        lstm_out, _ = self.lstm(batch)

        dense_1_out = self.dense_1(lstm_out.view(self.lstm_len, -1))

        dense_2_out = self.dense_2(dense_1_out.view(self.dense_1_len, -1))

        dense_3_out = self.dense_3(dense_2_out.view(self.dense_0_len, -1))

        reward_out = torch.nn.functional.log_softmax(dense_out, dim=1)

        return(reward_out)
