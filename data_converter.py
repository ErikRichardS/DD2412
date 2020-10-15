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
	#transforms.ToPILImage()
])



trainset = torchvision.datasets.CIFAR10(root="Data", train=False, download=False, transform=transform)


trn_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

path = "Data/trn_classes/class-"

path = "Data/tst_classes/class-"

for i in range(10):
	try:
		os.mkdir(path+str(i))
	except OSError as error:  
		print(error) 



class_counter = torch.zeros(10)

for (data, labels) in trn_loader:
	class_nr = labels.item()

	img = transforms.ToPILImage(mode="RGB")( torch.squeeze(data, dim=0) )
	img.save(path+str(class_nr)+"/img"+str(class_counter[class_nr].item())+".jpg")

	class_counter[class_nr] += 1


