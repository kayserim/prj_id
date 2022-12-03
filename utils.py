import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def compute_batch_accuracy(output, target):
	"""Computes the accuracy for a batch"""
	with torch.no_grad():

		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return correct * 100.0 / batch_size


#TODO if cvs file gets too large we will need to write custom dataset to load it in pieces
#TODO move this to utils and also refer in train and evaluate
def load_dataset(path, model_type, hours_limit):
	df = pd.read_csv(path)
	num_features = None
	target = torch.tensor(df.POSITIVE.to_numpy(), dtype=torch.long)
	df = df.drop(['POSITIVE'], axis=1)

	if model_type in ['LSTM']:
		df = df.loc[:,df.columns.str.startswith('HRLY')]#only using hourly data as per paper	
		x = torch.tensor(df.to_numpy(), dtype=torch.float32)
		x_dim1, x_dim2 = x.shape  
		num_features = hours_limit
		data = x.reshape((x_dim1, x_dim2//num_features, num_features))
	elif model_type in ['LSTMCNN']:
		df = df.loc[:,~df.columns.str.startswith('CESTAT')]# not using bias and rate data as per paper		
		x = torch.tensor(df.to_numpy(), dtype=torch.float32)
		x_dim1, x_dim2 = x.shape  
		# it is not clear in the paper how non-hourly features are sequenced hence will zero pad to make data to look like all hourly data.
		hourly_columns = sum(df.columns.str.startswith('HRLY'))
		hourly_features = hourly_columns//hours_limit
		zero_pad_size = hourly_features*(1+((x_dim2-1)//hourly_features))-x_dim2 #first part is ceil division
		num_features = (x_dim2 + zero_pad_size)//hourly_features
		zero_pad = torch.zeros(x_dim1,zero_pad_size)
		x_padded = torch.cat([x,zero_pad], dim = 1) 
		data = x_padded.reshape((x_dim1, hourly_features, num_features))
	else:
		df = df.loc[:,~df.columns.str.startswith('HRLY')]#for LR not using hourly data as per paper
		x = torch.tensor(df.to_numpy(), dtype=torch.float32)
		x_dim1, x_dim2 = x.shape
		num_features = x_dim2
		data = x

	dataset = TensorDataset(data, target)
	return num_features, dataset
	
def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	plotting = {'Loss': {'tra_data': train_losses, 'val_data': valid_losses},
				'Accuracy': {'tra_data': train_accuracies, 'val_data': valid_accuracies}}
	for type, data in plotting.items():
		plt.plot(data['tra_data'], label='training')
		plt.plot(data['val_data'], label='validation')
		plt.title(type + " Curve")
		plt.xlabel("Epoch")
		plt.ylabel(type)
		plt.grid()
		plt.legend(loc="best")
		plt.show() #todo save instead?
		#plt.savefig(type + "_curve")
		plt.close()
		
def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results

