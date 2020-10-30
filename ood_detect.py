import torch 

from loss import entropy



# Function of the OOD detection algorithm 
def ood_detect_score(classifier, data, epsilon=0.002, temperature=1000):
	data.requires_grad = True
	output = classifier( data, temp=temperature )
	loss = entropy(output)
	grad = torch.sign( torch.autograd.grad(loss, data)[0] )

	data_perturbed = data - epsilon*grad

	output_perturbed = classifier( data_perturbed, temp=temperature ).detach()

	return torch.max(output_perturbed, dim=-1)[0] + torch.sum( output_perturbed * torch.log(output_perturbed), dim=-1 )
	


def ood_detection(data, classifier_ensemble, conversion_matrises, nr_classes, temperature=1000):
	#data = torch.unsqueeze( transforms.ToTensor()(img).cuda(), dim=0 )
	class_score = torch.zeros(data.shape[0], nr_classes)
	ood_score = 0

	for i, classifier in enumerate(classifier_ensemble):
		class_score += torch.squeeze( torch.mm( classifier(data).detach(), conversion_matrises[i] ) ).cpu()
		
		ood_score += ood_detect_score(classifier, data, temperature=temperature)


	return class_score, ood_score
