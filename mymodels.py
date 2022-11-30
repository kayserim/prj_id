import torch
import torch.nn as nn

NUM_OUTPUTS = 2 # as per paper
class MyLR(nn.Module):
	def __init__(self, num_features):
		super(MyLR, self).__init__()
		self.fc = nn.Linear(num_features, NUM_OUTPUTS)

	def forward(self, x):
		return self.fc(x)

#TODO temporary - need to match the paper
class MyLSTM(nn.Module):
	def __init__(self,num_features, config=None):
		super(MyLSTM, self).__init__()
		class default_config:
		    LSTM_HIDDEN_SIZE = 32
		    LSTM_NUM_LAYERS = 3
		    LSTM_DROPOUT = 0.1
		self.config = config if config is not None else default_config# for backwards compatibility
		print(self.config)
		self.lstm = nn.LSTM(input_size=num_features, hidden_size=self.config.LSTM_HIDDEN_SIZE, num_layers=self.config.LSTM_NUM_LAYERS, dropout=self.config.LSTM_DROPOUT, bidirectional=True, batch_first=True)
		self.fc = nn.Linear(self.config.LSTM_HIDDEN_SIZE*2, NUM_OUTPUTS)  

	def forward(self, x):
		out, (h,c) = self.lstm(x)
		out = self.fc(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1) )#bi-directional need to be handled differently
		return out
