from default_values import default_values
from validate import validate_check
from make_directory import make_directory
from cuda_check import cuda_check
from make_models import make_models
from reading_data import reading_data
from train_GAN_pytorch import train_GAN as train_GAN_pytorch
from train_GAN_keras import train_GAN as train_GAN_keras
from save_models import save_models
from evaluate import evaluate
from argument_parser import argument_parser
from time import time

if __name__ == "__main__":
	"""Driver function."""
	start = time()
	conf_data = argument_parser()
	validate_check(conf_data)
	configured_parameters = default_values(conf_data)
	print (configured_parameters)
	configured_parameters = make_directory(configured_parameters)
	configured_parameters = cuda_check(configured_parameters)
	configured_parameters = make_models(configured_parameters)
	configured_parameters = reading_data(configured_parameters)
	setting_up_time = time() -  start
	time_step_2 = time()
	
	if conf_data['backend'] == 0: #0 for PyTorch
		configured_parameters = train_GAN_pytorch(configured_parameters)

	if conf_data['backend'] == 1: #1 for keras
		configured_parameters = train_GAN_keras(configured_parameters)

	training_time = time() - time_step_2
	configured_parameters = save_models(configured_parameters)
	if conf_data['GAN_model']['seq'] == 0:
		if conf_data['backend'] == 0:
			score = evaluate(conf_data)
		elif conf_data['backend'] == 1:
			print ("Evaluation metric yet to be made compatible with keras")
			score = -1


	log_file = conf_data['log_file']
	log_file.write(" Time of training (Avg over epochs) = {} \n".format(float(training_time/float(conf_data['GAN_model']['epochs']))))
	log_file.write(" Time of setting_up = {} \n".format(setting_up_time))
	log_file.close()
	print (" Time of training (Avg over epochs) = {} \n".format(float(training_time/float(conf_data['GAN_model']['epochs']))))
	if conf_data['GAN_model']['seq'] == 0:
		print (conf_data['metric_evaluate']+" score of the trained GAN is = {}".format(score))
