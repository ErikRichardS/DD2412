import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


from time import time
import numpy as np
from random import shuffle
import os

from vgg16 import VGG16
from densenet import DenseNet
from loss import *
from data_manager import *



def create_labels(input_labels, nr_classes, batch_size):
	labels = torch.zeros(batch_size, nr_classes)
	for i in range(batch_size):
		labels[i, input_labels[i].item()] = 1

	return labels.cuda()


def ood_detect_score(classifier, data, epsilon=0.002, temperature=1000):
	data.requires_grad = True
	output = classifier( data, temp=temperature )
	loss = entropy(output)
	grad = torch.sign( torch.autograd.grad(loss, data)[0] )

	data_perturbed = data - epsilon*grad

	output_perturbed = classifier( data_perturbed, temp=temperature ).detach()

	return torch.max(output_perturbed, dim=-1)[0] + torch.sum( output_perturbed * torch.log(output_perturbed), dim=-1 )
	



def test_classifier(net, id_loader, ood_loader, threshold=0.6):
	correct = 0
	total_id = 0
	ood_false = 0

	net = net.eval()

	for (data, labels) in id_loader:
		id_data = data.cuda()
		labels =  torch.squeeze(labels.cuda())

		outputs = net(id_data).detach()

		id_pos = torch.argmax(outputs, dim=-1) == labels
		correct += torch.sum(id_pos)
		total_id += len(labels)

		ood_score = ood_detect_score(net, id_data)
		ood_false += torch.sum(ood_score) / len(labels)


	total_ood = 0
	ood_true = 0

	for (data, labels) in ood_loader:
		ood_data = data.cuda()

		outputs = net(ood_data).detach()

		ood_score = ood_detect_score(net, ood_data)
		ood_true += torch.sum( torch.sum(ood_score < 0.5, dim=-1) == 10 )

		total_ood += len(labels)

		
	print(ood_false)
	print(ood_true)
	#print( float(ood_true) / float(total_ood) )

	return float(correct) / float(total_id),  0.0 #float(ood_true) / float(total_ood) 


def save_checkpoint(epoch, net, optimizer, ood_accuracy):
	torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ood_accuracy' : ood_accuracy,
            }, "checkpoint.pt")

def load_checkpoint(net, optimizer):
	checkpoint = torch.load("checkpoint.pt")
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	ood_accuracy = checkpoint['ood_accuracy']

	return epoch, net, optimizer, ood_accuracy

def checkpoint_exists():
	return os.path.isfile("checkpoint.pt")


def train_classifier(k, class_list):


	ood_classes = class_list[ int(k*2):int(k*2+2) ]
	id_classes = [x for x in class_list if x not in ood_classes]
	nr_classes = len(class_list)


	# Hyper Parameters
	num_epochs = 100
	batch_size = 80
	#learning_rate = 0.01
	learning_rate = np.linspace( 0.1, 0.0001, num_epochs )
	w_decay = 0.0005
	learning_decay = 0.9


	net = DenseNet(growthRate=12, depth=100, reduction=0.5, nClasses=nr_classes, bottleneck=True)


	trn_id_dataset = ImageDataset(exclude=ood_classes) # Training data
	trn_ood_dataset = ImageDataset(exclude=id_classes)

	vld_id_dataset = ImageDataset(exclude=ood_classes, train=False)
	vld_ood_dataset = ImageDataset(exclude=id_classes, train=False)


	# Loaders handle shufflings and splitting data into batches
	trn_id_loader = torch.utils.data.DataLoader(trn_id_dataset, batch_size=batch_size, shuffle=True)
	trn_ood_loader = torch.utils.data.DataLoader(trn_ood_dataset, batch_size=int(batch_size/4), shuffle=True)

	vld_id_loader = torch.utils.data.DataLoader(vld_id_dataset, batch_size=batch_size, shuffle=True)
	vld_ood_loader = torch.utils.data.DataLoader(vld_ood_dataset, batch_size=int(batch_size/4), shuffle=True)


	# Criterion calculates the error/loss of the output
	criterion = EMLoss()
	# Optimizer does the backprop to adjust the weights of the NN
	optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate[0], momentum=0.9, weight_decay=w_decay) 
	

	best_ood_accuracy = 0
	print("Begin training...")

	epoch_start = 0
	if checkpoint_exists():
		epoch_start, net, optimizer, best_ood_accuracy = load_checkpoint(net, optimizer)


	print("Starting at epoch %d" % epoch_start)
	

	for epoch in range(epoch_start, num_epochs):
		save_checkpoint(epoch, net, optimizer, best_ood_accuracy)

		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate[epoch]
		
		#print(optimizer.lr)

		t1 = time()

		loss_sum = 0

		ood_iterator = iter(trn_ood_loader)

		net = net.train()

		for i, (data, labels) in enumerate(trn_id_loader):
			# Load data into GPU using cuda
			id_data = data.cuda()
			#labels = labels.cuda()
			labels = create_labels(labels, nr_classes, len(labels))


			ood_data = next(ood_iterator)[0].cuda()

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			id_outputs = net(id_data)
			ood_outputs = net(ood_data)

			loss = criterion(id_outputs, ood_outputs, labels)
			loss.backward()
			optimizer.step()

			loss_sum += loss

		
		id_accuracy, ood_accuracy = test_classifier(net, vld_id_loader, vld_ood_loader)

		t2 = time()


		if id_accuracy > 0.98 and best_ood_accuracy < ood_accuracy:
			best_ood_accuracy = ood_accuracy
			torch.save(net, "loc" + str(k) + ".pt")

		print("Epoch : %d \t Time : %0.3f m \t Loss : %0.3f \t ID Accuracy %0.4f \t OOD Accuracy %0.4f" % ( epoch , (t2-t1)/60 , loss_sum , id_accuracy , ood_accuracy))

		



def ood_detection(img, classifier_ensemble, nr_classes):
	data = torch.unsqueeze( transforms.ToTensor()(img).cuda(), dim=0 )

	class_score = torch.zeros(nr_classes)
	ood_score = 0

	for classifier in classifier_ensemble:
		class_score += classifier( data )

		
		ood_score += ood_detect_score(classifier, data)


	return class_score, ood_score


K = 5

#classes = [i for i in range(10)]
#shuffle(classes)
#torch.save(torch.tensor(classes), "class_list.pt")
classes = [i.item() for i in torch.load("class_list.pt")]

for k in range(K):

	train_classifier(k, classes)
