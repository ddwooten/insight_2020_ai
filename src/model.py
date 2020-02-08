# This file builds the model class as requested by PyTorch
# Credit to the PyTorch team and the excellent tutorial found at pytorch.org

import pytorch as torch

class ActorModel(torch.nn.Module):

    def __init__(self, feature_len, lstm_len, dense_len):

        super(ActorModel, self).__init__()

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
