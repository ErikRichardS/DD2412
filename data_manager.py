import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

import os
from PIL import Image
import random


img_size = 256



#transform = transforms.Compose([
#	transforms.Resize((img_size,img_size)),
#	transforms.ToTensor()
#])


def get_training_data():


	dataset = ImageDataset("Data/img_train_shape", "Data/img_train_skeleton")

	return dataset





def set_ood(label):
	label = (label / label) * 0.1

	return label



# Dataset for handling the image input and output. 
# Takes the directory to the input and output files. 
# Requires the input and output files to have the same names.
class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, train_directory, label_directory):
		self.trn_dir = train_directory
		self.lbl_dir = label_directory
		self.file_list = os.listdir( train_directory)
		self.file_list.remove(".DS_Store")


	def __getitem__(self, idx):
		transform = get_random_transform()

		# Load the img and turn it into a Torch tensor matrix
		link = self.trn_dir+"/"+self.file_list[idx]
		data = transform( Image.open(link) )


		# Create label
		link = self.lbl_dir+"/"+self.file_list[idx]
		label = transform( Image.open(link) )

		return (data, label)


	def __len__(self):
		return len(self.file_list)