import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import torch
import numpy as np

def reading_data(conf_data):
	"""Reading input data and creating dataloader.
	
	Parameters
	----------
	conf_data: dict
		Dictionary containing all parameters and objects. 		

	Returns
	-------
	conf_data: dict
		Dictionary containing all parameters and objects. 		

	"""
	if conf_data['backend'] == 0:
		if conf_data['GAN_model']['seq'] == 0:
			g_input_shape = int(conf_data['generator']['input_shape'])
			mini_batch_size = int(conf_data['GAN_model']['mini_batch_size'])


			#Loading the dataset.
			dataset = pickle.load( open( conf_data['data_path'], "rb" ) )
			#print ("This is the dataset",conf_data['data_path'])
			#print (dataset.shape)
			#exit()
			if conf_data['GAN_model']['data_label'] == 1:
				dataset_img = dataset[0]
				dataset_label = dataset[1]

				#dataset_img = dataset_img[:100]
				#dataset_label = dataset_label[:100]
			else:
				dataset_img = dataset
				#dataset_img = dataset_img[:100]
			# print ("here")
			# print (dataset.shape)

			
			transform=transforms.Compose([transforms.Resize(g_input_shape), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #Transformation function to normalise the iamge pixel values within the range -1 and 1, 
			transforms_data = []
			if conf_data['GAN_model']['data_label'] == 1:
				for i,j in zip(dataset_img,dataset_label):
					x = transform(Image.fromarray(i, mode='L'))
					transforms_data.append((x,j))
			else:
				for i in dataset_img:	
					if conf_data['generator']['channels'] == 1:
						x = transform(Image.fromarray(i, mode='L'))
					elif conf_data['generator']['channels'] == 3:
						x = transform(Image.fromarray(i, mode='RGB'))
					transforms_data.append(x)
			conf_data['transformed'] = transforms_data

			loading = transforms_data
			dataloader = torch.utils.data.DataLoader(
				loading,
				batch_size=mini_batch_size, shuffle=True)

			conf_data['dataloader_size'] = len(dataloader)

			conf_data['data_learn'] = dataloader

	elif conf_data['backend'] == 1:
		# print ("Here is for 1")
		data_label = int(conf_data['GAN_model']['data_label'])
		
		if data_label == 0:
			dataset = pickle.load( open( conf_data['data_path'], "rb" ) )
		elif data_label == 1:
			dataset,lables = pickle.load( open( conf_data['data_path'], "rb" ) )
		dataset = dataset.reshape([-1,conf_data['generator']['input_shape'],conf_data['discriminator']['input_shape'],conf_data['generator']['channels']])
		conf_data['dataloader_size'] = len(dataset)
		if data_label == 1:
			dataset = (dataset,lables)
		conf_data['data_learn'] = dataset

	return conf_data
