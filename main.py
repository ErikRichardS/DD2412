import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


from time import time
import numpy as np
import os

from densenet3 import DenseNet3
from loss import *
from data_manager import *




def test_classifier(net, id_loader, ood_loader, min_id_accuracy, threshold=0.6):
	correct = 0
	total_id = 0
	ood_false = 0

	net = net.eval()

	for (data, labels) in id_loader:
		id_data = data.cuda()
		labels =  torch.squeeze(labels.cuda())

		outputs = net(id_data).detach()

		id_pos = torch.argmax(outputs, dim=-1) == torch.argmax(labels, dim=-1) 
		correct += torch.sum(id_pos)
		total_id += len(labels)

		#ood_score = ood_detect_score(net, id_data)
		ood_false += torch.sum( torch.sum(outputs < 0.5, dim=-1) == 8 )


	total_ood = 0
	ood_true = 0

	id_accuracy = float(correct) / float(total_id)

	if id_accuracy < min_id_accuracy:
		return id_accuracy, 0.0, float(ood_false) / float(total_id) 

	for (data, labels) in ood_loader:
		ood_data = data.cuda()

		outputs = net(ood_data).detach()

		#ood_score = ood_detect_score(net, ood_data)
		ood_true += torch.sum( torch.sum(outputs < 0.5, dim=-1) == 8 )

		total_ood += len(labels)

		
	print(ood_false)
	print(ood_true)
	#print( float(ood_true) / float(total_ood) )

	return id_accuracy,  float(ood_true) / float(total_ood),  float(ood_false) / float(total_id) 



def save_checkpoint(epoch, net, optimizer, id_accuracy, ood_accuracy):
	torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'id_accuracy' : id_accuracy,
            'ood_accuracy' : ood_accuracy,
            }, "checkpoint.pt")

def load_checkpoint(net, optimizer):
	checkpoint = torch.load("checkpoint.pt")
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	id_accuracy = checkpoint['id_accuracy']
	ood_accuracy = checkpoint['ood_accuracy']


	return epoch, net, optimizer, id_accuracy, ood_accuracy

def checkpoint_exists():
	return os.path.isfile("checkpoint.pt")

def delete_checkpoint():
	os.remove("checkpoint.pt") 


def train_classifier(id_classes, ood_classes):
	# Set id and ood classes
	nr_classes = len(id_classes)

	conversion_matrix = create_conversion_matrix(nr_classes+len(ood_classes), id_classes) 

	# Hyper Parameters
	num_epochs = 100
	id_batch_size = 50
	ood_batch_size = 50
	learning_rate = np.linspace( 0.1, 0.0001, num_epochs )
	w_decay = 0.0005
	delta_accuracy = 0.02


	#net = DenseNet(growthRate=12, depth=100, reduction=0.5, nClasses=nr_classes, bottleneck=True)
	net = DenseNet3(depth=100, num_classes=nr_classes)


	trn_id_dataset = ImageDataset(include=id_classes, conv_mat=conversion_matrix) # Training data
	trn_ood_dataset = ImageDataset(include=ood_classes)

	vld_id_dataset = ImageDataset(include=id_classes, conv_mat=conversion_matrix, train=False)
	vld_ood_dataset = get_SUN_dataset()


	# Loaders handle shufflings and splitting data into batches
	trn_id_loader = torch.utils.data.DataLoader(trn_id_dataset, batch_size=id_batch_size, shuffle=True)
	trn_ood_loader = torch.utils.data.DataLoader(trn_ood_dataset, batch_size=ood_batch_size, shuffle=True)


	vld_id_loader = torch.utils.data.DataLoader(vld_id_dataset, batch_size=id_batch_size, shuffle=False)
	vld_ood_loader = torch.utils.data.DataLoader(vld_ood_dataset, batch_size=ood_batch_size, shuffle=False)


	# Criterion calculates the error/loss of the output
	criterion = EMLoss()
	# Optimizer does the backprop to adjust the weights of the NN
	optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate[0], momentum=0.9, weight_decay=w_decay) 
	

	best_ood_accuracy = 0
	best_id_accuracy = 0
	print("Begin training...")

	epoch_start = 0
	if checkpoint_exists():
		epoch_start, net, optimizer, best_id_accuracy, best_ood_accuracy = load_checkpoint(net, optimizer)


	print("Starting at epoch %d" % epoch_start)
	

	for epoch in range(epoch_start, num_epochs):
		save_checkpoint(epoch, net, optimizer, best_id_accuracy, best_ood_accuracy)

		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate[epoch]


		t1 = time()

		loss_sum = 0

		ood_iterator = iter(trn_ood_loader)

		# Set network to training mode
		net = net.train()

		for i, (data, labels) in enumerate(trn_id_loader):
			# Load data into GPU using cuda
			id_data = data.cuda()
			labels = labels.cuda()


			ood_data = None
			try:
				ood_data = next(ood_iterator)[0].cuda()
			except:
				ood_iterator = iter(trn_ood_loader)
				ood_data = next(ood_iterator)[0].cuda()


			# Forward + Backward + Optimize
			optimizer.zero_grad()
			id_outputs = net(id_data)
			ood_outputs = net(ood_data)

			loss = criterion(id_outputs, ood_outputs, labels)
			loss.backward()
			optimizer.step()

			loss_sum += loss

		
		id_accuracy, ood_true_accuracy, ood_false_accuracy = test_classifier(net, vld_id_loader, vld_ood_loader, best_id_accuracy-delta_accuracy)

		t2 = time()

		# If ID accuracy high enough and OOD accuracy the new best, or ID accuracy is simply more than 2% better than the last
		if (id_accuracy > best_id_accuracy-delta_accuracy and best_ood_accuracy < ood_true_accuracy) or (id_accuracy > best_id_accuracy+delta_accuracy):
			best_ood_accuracy = ood_true_accuracy
			torch.save(net, "loc" + str(k) + ".pt")

			if id_accuracy > best_id_accuracy:
				best_id_accuracy = id_accuracy
			

		print("Epoch : %d \t Time : %0.3f m \t Loss : %0.3f \t ID Accuracy %0.4f \t OOD True Accuracy %0.4f \t OOD False Error Rate %0.4f" % ( epoch , (t2-t1)/60 , loss_sum , id_accuracy , ood_true_accuracy, ood_false_accuracy))

	print("Delete checkpoint")
	delete_checkpoint()






class_partitions = []

if os.path.isfile("partitions.pt"):
	class_partitions = torch.load("partitions.pt")
else:
	class_partitions = split_classes(10)
	torch.save(class_partitions, "partitions.pt")


K = 5
k_start = 0
if os.path.isfile("classifier_nr.pt"):
	k_start = torch.load("classifier_nr.pt")


for k in range(k_start,K):
	torch.save(k, "classifier_nr.pt")

	id_classes, ood_classes = get_id_ood_partitions(class_partitions, k)

	train_classifier(id_classes, ood_classes)

os.remove("classifier_nr.pt") 