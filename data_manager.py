import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms

import os
from PIL import Image
import random





# During training, we augment our training data with random flip and random cropping.
"""
transform_trn = transforms.Compose([
	transforms.RandomCrop(32, padding=4, padding_mode="constant"),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomVerticalFlip(p=0.4),
	transforms.ToTensor()
])
"""


def set_ood(label):
	label = (label / label) * 0.1

	return label



# Dataset for handling the image input and output. 
# Takes the directory to the input and output files. 
# Requires the input and output files to have the same names.
class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, exclude=[], train=True):
		directory = "Data/trn_dataset"
		if not train:
			directory = "Data/tst_dataset"

		self.file_list = []
		self.label_list =[]

		for i in range(10):
			if i in exclude:
				#print(i)
				continue

			subdir = directory+"/class-"+str(i)
			#print(subdir)
			img_list = os.listdir(subdir)
			for img in img_list:
				#if int(subdir[-1]) != i:
				#	print("Error")

				self.file_list.append(subdir+"/"+img)
				self.label_list.append(i)

		#print(self.label_list)

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

		label = self.label_list[idx]

		#if int(link[23]) != label:
		#	print("Error")

		return (data, label)


	def __len__(self):
		return len(self.file_list)






def get_SUN_dataset(dataset="iSUN"):
	if dataset == "iSUN":
		return torchvision.datasets.ImageFolder("Data/iSUN", transform=transforms.ToTensor())

	elif dataset =="LSUN":
		return torchvision.datasets.ImageFolder("Data/LSUN", transform=transforms.ToTensor())

	else:
		return torchvision.datasets.ImageFolder("Data/LSUN_resize", transform=transforms.ToTensor())