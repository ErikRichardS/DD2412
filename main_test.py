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
from ood_detect import *





class_partitions = torch.load("partitions.pt")
nr_classes = 10
class_list = []
conversion_matrix = []
classifier_ensemble = []


for k in range(5):
	id_classes, ood_classes = get_id_ood_partitions(class_partitions, k)
	conv_mat = torch.transpose(create_conversion_matrix( nr_classes, id_classes), 0, 1).cuda()
	conversion_matrix.append( conv_mat )

	class_list = class_list + class_partitions[k]

	classifier_ensemble.append( torch.load("loc" + str(k) + ".pt") )



def calculate_ood_scores(loader, temp):
	#correct = 0
	total = len(loader)

	ood_total = 0
	for (data, labels) in loader:
		data = data.cuda()

		class_score, ood_score = ood_detection(data, classifier_ensemble, conversion_matrix, nr_classes, temp)
		#correct += torch.sum( torch.argmax(class_score, dim=-1) == torch.argmax(labels, dim=-1) ).item()
		ood_total += torch.sum(ood_score)


	return ood_total.item()


batch_size = 50

testset_name_1 = "LSUN"
testset_name_2 = "LSUN_resize"

vld_id_dataset = ImageDataset(include=class_list, train=False)
vld_ood_dataset_1 = get_SUN_dataset(testset_name_1)
vld_ood_dataset_2 = get_SUN_dataset(testset_name_2)


vld_id_loader = torch.utils.data.DataLoader(vld_id_dataset, batch_size=batch_size, shuffle=False)
vld_ood_loader_1 = torch.utils.data.DataLoader(vld_ood_dataset_1, batch_size=batch_size, shuffle=False)
vld_ood_loader_2 = torch.utils.data.DataLoader(vld_ood_dataset_2, batch_size=batch_size, shuffle=False)

temperature = [1, 10, 100, 1000]

id_score = torch.zeros(len(temperature))
ood_score_1 = torch.zeros(len(temperature))
ood_score_2 = torch.zeros(len(temperature))

for i, t in enumerate(temperature):
	print("T : %d" % t)
	av_id = calculate_ood_scores(vld_id_loader, t) / float(len(vld_id_dataset))
	av_ood_1 = calculate_ood_scores(vld_ood_loader_1, t) / float(len(vld_ood_dataset_1))
	av_ood_2 = calculate_ood_scores(vld_ood_loader_2, t) / float(len(vld_ood_dataset_2))
	print(av_id)
	print(av_ood_1)
	print(av_ood_2)

	id_score[i] = av_id
	ood_score_1[i] = av_ood_1
	ood_score_2[i] = av_ood_2


torch.save(id_score, "id_score.pt")
torch.save(ood_score_1, testset_name_1 + "_ood_score.pt")
torch.save(ood_score_2, testset_name_2 + "_ood_score.pt")