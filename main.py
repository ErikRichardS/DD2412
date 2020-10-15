import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.transforms.functional as F

from time import time
import numpy as np

from vgg16 import VGG16
from densenet import DenseNet
from loss import EMLoss
from data_manager import *



def create_labels(input_labels, batch_size):
	labels = torch.zeros(batch_size, nr_classes)
	for i in range(batch_size):
		labels[i, input_labels[i].item()] = 1

	return labels.cuda()



def train_classifier(keep_in, leave_out):
	# Hyper Parameters
	num_epochs = 100
	batch_size = 80
	learning_rate = 0.01
	#weight_decay = 1e-7
	learning_decay = 0.9


	net = DenseNet(growthRate=12, depth=100, reduction=0.5, nClasses=nr_classes, bottleneck=True)


	trn_id_dataset = ImageDataset("Data/trn_classes", exclude=leave_out) # Training data
	trn_ood_dataset = ImageDataset("Data/trn_classes", exclude=keep_in)


	# Loaders handle shufflings and splitting data into batches
	trn_id_loader = torch.utils.data.DataLoader(trn_id_dataset, batch_size=batch_size, shuffle=True)
	trn_ood_loader = torch.utils.data.DataLoader(trn_id_dataset, batch_size=int(batch_size/4), shuffle=True)

	#vld_loader = torch.utils.data.DataLoader(vld_dataset, batch_size=batch_size)


	# Criterion calculates the error/loss of the output
	# Optimizer does the backprop to adjust the weights of the NN
	criterion = EMLoss()
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 



	for epoch in range(num_epochs):
		t1 = time()

		loss_sum = 0

		ood_iterator = iter(trn_ood_loader)

		for i, (data, labels) in enumerate(trn_id_loader):
			# Load data into GPU using cuda
			id_data = data.cuda()
			#labels = labels.cuda()
			labels = create_labels(labels, batch_size)


			ood_data = next(ood_iterator)[0].cuda()

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			id_outputs = net(id_data)
			ood_outputs = net(ood_data)

			loss = criterion(id_outputs, ood_outputs, labels)
			loss.backward()
			optimizer.step()

			loss_sum += loss

		t2 = time()
		print("Epoch time : %0.3f m \t Loss : %0.3f" % ( (t2-t1)/60 , loss_sum ))

		torch.save(net, "skeleton_net.pt")






K = 5
nr_classes = 10
classes = [i for i in range(nr_classes)]

for i in range(K):
	ood_classes = classes[ i*2 : i*2+1 ]
	id_classes = [x for x in classes if x not in ood_classes]

	train_classifier(id_classes, ood_classes)
