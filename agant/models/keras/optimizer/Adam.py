from keras.optimizers import Adam

def optimizer_block(conf_data):
	d_b1 = float(conf_data['discriminator']['optimizer']['b1'])
	d_b2 = float(conf_data['discriminator']['optimizer']['b2'])
	optimizer = Adam(d_b1,d_b2)
	return optimizer