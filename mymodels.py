import torch
import torch.nn as nn

NUM_OUTPUTS = 2 # as per paper
class MyLR(nn.Module):
	def __init__(self, num_features):
		super(MyLR, self).__init__()
		self.fc = nn.Linear(num_features, NUM_OUTPUTS)

	def forward(self, x):
		return self.fc(x)

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

class MyLSTMCNN(nn.Module):
	def __init__(self,num_features, config=None):
		super(MyLSTMCNN, self).__init__()
		class default_config:
		    LSTM_HIDDEN_SIZE = 32
		    LSTM_NUM_LAYERS = 3
		    LSTM_DROPOUT = 0.1
		self.config = config if config is not None else default_config# for backwards compatibility
		print(self.config)
		self.lstm = nn.LSTM(input_size=num_features, hidden_size=self.config.LSTM_HIDDEN_SIZE, num_layers=self.config.LSTM_NUM_LAYERS, dropout=self.config.LSTM_DROPOUT, bidirectional=True, batch_first=True)
		
		FILTER1_SIZE, FILTER2_SIZE, FILTER3_SIZE = 2, 3, 4
		FILTER_STRIDE = 1
		MAXPOOL_SIZE = 2
		MAXPOOL_STRIDE = 2
		self.conv_stack1 = nn.Sequential(
			nn.Conv1d(in_channels=1, out_channels=1, kernel_size=FILTER1_SIZE, stride=FILTER_STRIDE, padding=0),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=MAXPOOL_SIZE, stride=MAXPOOL_STRIDE),
	    	nn.Flatten()
		)
		self.conv_stack2 = nn.Sequential(
			nn.Conv1d(in_channels=1, out_channels=1, kernel_size=FILTER2_SIZE, stride=FILTER_STRIDE, padding=0),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=MAXPOOL_SIZE, stride=MAXPOOL_STRIDE),
	    	nn.Flatten()
	    )
		self.conv_stack3 = nn.Sequential(
			nn.Conv1d(in_channels=1, out_channels=1, kernel_size=FILTER3_SIZE, stride=FILTER_STRIDE, padding=0),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=MAXPOOL_SIZE, stride=MAXPOOL_STRIDE),
            nn.Flatten()
		)

		A = self.config.LSTM_HIDDEN_SIZE*2
		B1 = ((((A-FILTER1_SIZE)//FILTER_STRIDE)+1)-MAXPOOL_SIZE)//MAXPOOL_STRIDE + 1
		B2 = ((((A-FILTER2_SIZE)//FILTER_STRIDE)+1)-MAXPOOL_SIZE)//MAXPOOL_STRIDE + 1
		B3 = ((((A-FILTER3_SIZE)//FILTER_STRIDE)+1)-MAXPOOL_SIZE)//MAXPOOL_STRIDE + 1
		self.fc = nn.Linear(B1+B2+B3, NUM_OUTPUTS)  

	def forward(self, x):
		out, (h,c) = self.lstm(x)
		out = torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)#bi-directional need to be handled differently
		out = torch.unsqueeze(out, dim=1)
		out1 = self.conv_stack1(out)
		out2 = self.conv_stack2(out)
		out3 = self.conv_stack3(out)
		out = torch.cat([out1, out2, out3], dim=1)
		out = self.fc(out)
		return out
