import torch 
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.transforms.functional as F

import os
import PIL


transform = transforms.Compose([
	transforms.ToTensor()
])


trn = False

trainset = torchvision.datasets.CIFAR10(root="Data", train=trn, download=False, transform=transform)


loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

path = "Data/trn_dataset/class-"

if not trn:
	path = "Data/tst_dataset/class-"

"""
for i in range(10):
	try:
		os.mkdir(path+str(i))
	except OSError as error:  
		print(error) 
"""

class_counter = torch.zeros(10)

for (data, labels) in loader:
	class_nr = labels.item()

	#img = transforms.ToPILImage(mode="RGB")( torch.squeeze(data, dim=0) )
	link = path+str(class_nr)+"/img"+str( int(class_counter[class_nr].item()) )+".png"
	#img.save(link)

	d = transform( PIL.Image.open(link) )

	if not torch.equal( torch.squeeze(data, dim=0), d ):
		print("Error")

	class_counter[class_nr] += 1


