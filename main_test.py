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

# Load in necessary information
for k in range(5):
	id_classes, ood_classes = get_id_ood_partitions(class_partitions, k)
	conv_mat = torch.transpose(create_conversion_matrix( nr_classes, id_classes), 0, 1).cuda()
	conversion_matrix.append( conv_mat )

	class_list = class_list + class_partitions[k]

	classifier_ensemble.append( torch.load("loc" + str(k) + ".pt") )



def calculate_ood_scores(loader, temp, n, is_id=False):
	
	ood_scores = torch.zeros(n)
	j = 0

	correct = 0

	for data, labels in loader:
		data = data.cuda()

		class_score, ood_score = ood_detection(data, classifier_ensemble, conversion_matrix, nr_classes, temp)

		if is_id:
			class_ind = torch.argmax(class_score, 1)
			labels = torch.argmax(labels, 1)

			correct += torch.sum( class_ind == labels)		

		l = len(labels)
		ood_scores[j:j+l] = ood_score.cpu()
		j += l


	return ood_scores, float(correct) / float(n)


batch_size = 50

testset_name_1 = "LSUN"
testset_name_2 = "LSUN_resize"

vld_id_dataset = ImageDataset(include=class_list, train=False)
vld_ood_dataset_1 = get_SUN_dataset(testset_name_1)
vld_ood_dataset_2 = get_SUN_dataset(testset_name_2)

id_len = len(vld_id_dataset)
ood1_len = len(vld_ood_dataset_1)
ood2_len = len(vld_ood_dataset_2)

vld_id_loader = torch.utils.data.DataLoader(vld_id_dataset, batch_size=batch_size, shuffle=False)
vld_ood_loader_1 = torch.utils.data.DataLoader(vld_ood_dataset_1, batch_size=batch_size, shuffle=False)
vld_ood_loader_2 = torch.utils.data.DataLoader(vld_ood_dataset_2, batch_size=batch_size, shuffle=False)

temperature = [1, 10, 100, 1000]

id_score = [torch.zeros(1) for i in range(len(temperature))]
ood_score_1 = [torch.zeros(1) for i in range(len(temperature))]
ood_score_2 = [torch.zeros(1) for i in range(len(temperature))]


for i, t in enumerate(temperature):
	print("T : %d" % t)
	av_id, accuracy = calculate_ood_scores(vld_id_loader, t, id_len, is_id=True)
	print("ID done")
	print("Correct %0.4f" % accuracy)

	av_ood_1, _ = calculate_ood_scores(vld_ood_loader_1, t, ood1_len)
	print("OOD 1 done") 
	av_ood_2, _ = calculate_ood_scores(vld_ood_loader_2, t, ood2_len)
	print("OOD 2 done") 


	id_score[i] = av_id
	ood_score_1[i] = av_ood_1
	ood_score_2[i] = av_ood_2


torch.save(id_score, "id_score.pt")
torch.save(ood_score_1, testset_name_1 + "_ood_score.pt")
torch.save(ood_score_2, testset_name_2 + "_ood_score.pt")