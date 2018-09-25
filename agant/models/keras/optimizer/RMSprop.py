from keras.optimizers import RMSprop

def optimizer_block(conf_data):
	learning_rate = float(conf_data['generator']['optimizer']['learning_rate'])
	optimizer = RMSprop()
	return optimizer