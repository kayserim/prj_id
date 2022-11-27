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
	def __init__(self,num_features):
		super(MyLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size=num_features, hidden_size=32, num_layers=3, dropout=0.1, bidirectional=True, batch_first=True)
		self.fc = nn.Linear(32*2, NUM_OUTPUTS)  

	def forward(self, x):
		out, (h,c) = self.lstm(x)
		out = self.fc(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1) )#bi-directional handled differently
		return out
