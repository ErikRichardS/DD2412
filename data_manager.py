import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms

import os
from PIL import Image
from random import shuffle







def create_conversion_matrix(nr_classes, id_classes):
	id_classes = torch.tensor(id_classes)

	conv_mat = torch.zeros(nr_classes, len(id_classes))

	for i in range(len(id_classes)):
		conv_mat[id_classes[i],i] = 1

	return conv_mat


def split_classes(nr_classes, K=5):
	class_list = [i for i in range(nr_classes)]
	shuffle(class_list)

	partitions = []
	partition_size = int( nr_classes / K )

	for k in range(K):
		partitions.append(class_list[ k*partition_size:(k+1)*partition_size ])

	return partitions


def get_id_ood_partitions(partitions, exclude):
	merged_list = []

	for i in range(len(partitions)):
		if i == exclude:
			continue
		merged_list = merged_list + partitions[i]

	return merged_list, partitions[exclude]


def create_labels(input_labels):
	labels = torch.zeros(batch_size, nr_classes)
	for i in range(batch_size):
		labels[input_labels[i].item()] = 1

	return labels.cuda()


# Dataset for handling the image input and output. 
# Takes the directory to the input and output files. 
# Requires the input and output files to have the same names.
class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, include, conv_mat=torch.eye(10), train=True, cifar="cifar10"):
		directory = "Data/trn_dataset"
		if not train:
			directory = "Data/tst_dataset"

		self.class_list = include
		self.file_list = []
		self.label_list = []
		self.conv_mat = conv_mat


		for i in range(10):
			# Skip classes not part of dataset
			if i not in include:
				continue

			subdir = directory+"/class-"+str(i)
			img_list = os.listdir(subdir)
			for img in img_list:
				self.file_list.append(subdir+"/"+img)
				self.label_list.append(i)

		# Set the transform used for the data
		if train:
			self.transform = transforms.Compose([
				transforms.RandomCrop(32, padding=6, padding_mode="constant"),
				transforms.RandomHorizontalFlip(p=0.5),
				#transforms.RandomVerticalFlip(p=0.4),
				transforms.ToTensor()
			])
		else:
			self.transform = transforms.ToTensor()


	def __getitem__(self, idx):
		# Load the img and turn it into a Torch tensor matrix
		link = self.file_list[idx]
		data = self.transform( Image.open(link) )

		# Make the label as a one hot
		label = torch.zeros(1,10)
		label[ 0, self.label_list[idx] ] = 1
		label = torch.squeeze(torch.mm(label, self.conv_mat), dim=0)

		return (data, label)


	def __len__(self):
		return len(self.file_list)



def get_SUN_dataset(dataset="iSUN"):
	if dataset == "iSUN":
		return torchvision.datasets.ImageFolder("Data/iSUN", transform=transforms.ToTensor())

	elif dataset =="LSUN":
		return torchvision.datasets.ImageFolder("Data/LSUN", transform=transforms.Compose([ transforms.CenterCrop(32), transforms.ToTensor() ]) )

	else:
		return torchvision.datasets.ImageFolder("Data/LSUN_resize", transform=transforms.ToTensor())