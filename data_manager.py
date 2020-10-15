import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

import os
from PIL import Image
import random





# During training, we augment our training data with random flip and random cropping.

transform = transforms.Compose([
	transforms.ToTensor()
])




def set_ood(label):
	label = (label / label) * 0.1

	return label



# Dataset for handling the image input and output. 
# Takes the directory to the input and output files. 
# Requires the input and output files to have the same names.
class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, directory, exclude=[]):
		classes = os.listdir(directory)

		self.file_list = []
		self.class_list = []
		excluded_nr = 0

		for i in range(10):
			if i in exclude:
				excluded_nr += 1
				continue

			self.class_list.append(directory+"/"+classes[i])
			self.file_list = self.file_list +  os.listdir(directory+"/"+classes[i])

		self.files_per_class = int( len(self.file_list) / excluded_nr )


	def __getitem__(self, idx):
		# Load the img and turn it into a Torch tensor matrix
		class_nr = int( idx / self.files_per_class )

		link = self.class_list[ class_nr ] + "/" + self.file_list[idx]
		data = transform( Image.open(link) )

		label = torch.tensor([class_nr])

		return (data, label)


	def __len__(self):
		return len(self.file_list)