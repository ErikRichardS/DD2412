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
from data_manager import *









# Hyper Parameters
num_epochs = 100
batch_size = 20
learning_rate = 1e-4
#weight_decay = 1e-7
learning_decay = 0.9


transform = transforms.Compose([
	transforms.ToTensor()
])


trainset = torchvision.datasets.CIFAR10(root="Data", train=True, download=True, transform=transform)


net = DenseNet
trn_dataset = get_training_data() # Training data
vld_dataset = None # Validation data




# Loaders handle shufflings and splitting data into batches
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
#vld_loader = torch.utils.data.DataLoader(vld_dataset, batch_size=batch_size)


# Criterion calculates the error/loss of the output
# Optimizer does the backprop to adjust the weights of the NN
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) 



for epoch in range(num_epochs):
	t1 = time()

	loss_sum = 0

	for i, (data, labels) in enumerate(trn_loader):
		# Load data into GPU using cuda
		data = data.cuda()
		labels = labels.cuda()



		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = net(data)

		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		loss_sum += loss

	t2 = time()
	print("Epoch time : %0.3f m \t Loss : %0.3f" % ( (t2-t1)/60 , loss_sum ))

	torch.save(net, "skeleton_net.pt")