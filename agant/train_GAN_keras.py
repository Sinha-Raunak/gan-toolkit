import torch
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from training_function_generator_keras import training_function_generator
from compute_gradient_penalty import compute_gradient_penalty
from sample_image import sample_image
import matplotlib.pyplot as plt
import sys
import numpy as np 
"""
For keras model
"""

def sample_images(conf_data, epoch):
    latent_dim = int(conf_data['generator']['latent_dim'])
    generator = generator = conf_data['generator_model']

    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(conf_data['result_path']+"%d.png" % epoch)
    plt.close()

def sample_images_cgan(conf_data, epoch):
    generator = generator = conf_data['generator_model']
    r, c = 2, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.arange(0, 10).reshape(-1, 1)

    gen_imgs = generator.predict([noise, sampled_labels])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(conf_data['result_path']+"%d.png" % epoch)
    plt.close()

def train_GAN(conf_data):

    """Training Process for GAN.
    
    Parameters
    ----------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    Returns
    -------
    conf_data: dict
        Dictionary containing all parameters and objects.       

    """
    seq = conf_data['GAN_model']['seq']
    if seq == 1:
        pre_epoch_num = conf_data['generator']['pre_epoch_num']
        GENERATED_NUM = 10000 
        EVAL_FILE = 'eval.data'
        POSITIVE_FILE = 'real.data'
        NEGATIVE_FILE = 'gene.data'
    temp = 1 #TODO Determines how many times is the discriminator updated. Take this as a value input
    epochs = int(conf_data['GAN_model']['epochs'])
    # if seq == 0:
    #     data_train = conf_data['data_learn']
    mini_batch_size = int(conf_data['GAN_model']['mini_batch_size'])
    data_label = int(conf_data['GAN_model']['data_label'])
    cuda = conf_data['cuda']
    g_latent_dim = int(conf_data['generator']['latent_dim'])
    classes = int(conf_data['GAN_model']['classes'])

    w_loss = int(conf_data['GAN_model']['w_loss'])

    clip_value = float(conf_data['GAN_model']['clip_value'])
    n_critic = int(conf_data['GAN_model']['n_critic'])

    lambda_gp = int(conf_data['GAN_model']['lambda_gp'])

    log_file = open(conf_data['performance_log']+"/log.txt","w+")
    

    conf_data['epochs'] = epochs

    #print ("Just before training")
    if seq == 1: #TODO: Change back to 1 
        target_lstm = TargetLSTM(conf_data['GAN_model']['vocab_size'], conf_data['generator']['embedding_dim'], conf_data['generator']['hidden_dim'], conf_data['cuda'])
        if cuda == True:
            target_lstm = target_lstm.cuda()
        conf_data['target_lstm'] = target_lstm
        gen_data_iter = GenDataIter('real.data', mini_batch_size)
        generator = conf_data['generator_model']
        discriminator = conf_data['discriminator_model']
        g_loss_func = conf_data['generator_loss']
        d_loss_func = conf_data['discriminator_loss']
        optimizer_D = conf_data['discriminator_optimizer']
        optimizer_G = conf_data['generator_optimizer']

        for epoch in range(pre_epoch_num): #TODO: Change the range
            loss = train_epoch(generator, gen_data_iter, g_loss_func, optimizer_G,conf_data,'g')
            print('Epoch [%d] Model Loss: %f'% (epoch, loss))
            generate_samples(generator, mini_batch_size, GENERATED_NUM, EVAL_FILE,conf_data)
            eval_iter = GenDataIter(EVAL_FILE, mini_batch_size)
            loss = eval_epoch(target_lstm, eval_iter, g_loss_func,conf_data)
            print('Epoch [%d] True Loss: %f' % (epoch, loss))

        dis_criterion = d_loss_func
        dis_optimizer = optimizer_D

        #print('Pretrain Dsicriminator ...')
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, mini_batch_size)
        for epoch in range(5): #TODO: change back 5
            generate_samples(generator, mini_batch_size, GENERATED_NUM, NEGATIVE_FILE,conf_data)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, mini_batch_size)
            for _ in range(3): #TODO: change back 3
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer,conf_data,'d')
                print('Epoch [%d], loss: %f' % (epoch, loss))
        conf_data['generator_model'] = generator
        conf_data['discriminator_model'] = discriminator
        torch.save(conf_data['generator_model'].state_dict(),conf_data['save_model_path']+'/Seq/'+'pre_generator.pt')
        torch.save(conf_data['discriminator_model'].state_dict(),conf_data['save_model_path']+'/Seq/'+'pre_discriminator.pt') 
        
        conf_data['rollout'] = Rollout(generator, 0.8)
    """
    Actuall training loop
    """ 
    for epoch in range(epochs):
        conf_data['epoch'] = epoch
        if seq == 0:
            to_iter = [1]
        elif seq == 1: 
            to_iter = [1]

        for i, iterator in enumerate(to_iter):
            generator = conf_data['generator_model']
            discriminator = conf_data['discriminator_model']
            combined_model = conf_data['combined_model']

            conf_data['iterator'] = i 
            if seq == 0:
                if data_label == 1:
                    data_train,labels_train = conf_data['data_learn']

                elif data_label == 0:
                    data_train = conf_data['data_learn']

                # Adversarial ground truths
                
                valid = np.ones((mini_batch_size,1))
                conf_data['valid']=valid

                
                fake = np.zeros((mini_batch_size,1))
                conf_data['fake']=fake
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            if seq == 1:#TODO change this back to 1
                dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, mini_batch_size)
            if n_critic <= 0:
                n_critic = 1
            for i in range(n_critic): # TODO: Make this a parameter -> for x updates --> I am read the stored models here as well. Should I reamove this ???

                # Configure input
                idx = np.random.randint(0,data_train.shape[0],mini_batch_size)
                real_imgs = data_train[idx]

                if classes > 0:
                    labels = labels_train[idx]

                # Sample noise as generator input
                z = np.random.normal(0, 1, (mini_batch_size, g_latent_dim))
                

                if classes <= 0:
                    if seq == 0:
                        gen_imgs = generator.predict(z)
                    

                    if seq == 1:
                        generate_samples(generator,mini_batch_size,GENERATED_NUM,NEGATIVE_FILE,conf_data)
                        dis_data_iter = DisDataIter(POSITIVE_FILE,NEGATIVE_FILE,mini_batch_size)
                        loss = train_epoch(discriminator,dis_data_iter,d_loss_func,optimizer_D,conf_data,'d')
                        conf_data['d_loss'] = loss

                else:
                    if seq == 0:
                        gen_imgs =  generator.predict([z,labels])

                if seq == 0:
                    conf_data['gen_imgs'] = gen_imgs
                if seq == 0:
                    if w_loss == 0:
                        if classes <= 0:
                            real_loss = discriminator.train_on_batch(real_imgs,valid)
                            fake_loss = discriminator.train_on_batch(gen_imgs,fake)
                            d_loss = 0.5 * np.add(real_loss,fake_loss)
                        else:
                            real_loss = discriminator.train_on_batch([real_imgs, labels], valid)
                            fake_loss = discriminator.train_on_batch([gen_imgs,labels],fake)
                            d_loss = 0.5 * np.add(real_loss,fake_loss)

                    elif w_loss == 1:
                        real_loss = discriminator.train_on_batch(real_imgs,valid)
                        fake_loss = discriminator.train_on_batch(gen_imgs,fake)
                        d_loss =  np.subtract(fake_loss,real_loss)
                        if lambda_gp > 0:
                            conf_data['real_data_sample'] = real_imgs.data
                            conf_data['fake_data_sample'] = gen_imgs.data
                            conf_data = compute_gradient_penalty(conf_data)
                            gradient_penalty = conf_data['gradient_penalty']
                            d_loss = d_loss+ lambda_gp * gradient_penalty
                    conf_data['d_loss'] = d_loss

                if clip_value > 0:
                    for l in discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

            # -----------------   
            #  Train Generator
            # -----------------
            conf_data['generator_model'] = generator
            conf_data['discriminator_model'] = discriminator
            conf_data['combined_model'] = combined_model
            z = np.random.normal(0, 1, (mini_batch_size, g_latent_dim))

            if seq == 0:
                conf_data['noise'] = z

            training_function_generator(conf_data)
           

            if seq == 0:
                batches_done = epoch * len(data_train) + i
                if batches_done % int(conf_data['sample_interval']) == 0:
                    if classes <= 0:
                        sample_images(conf_data, epoch)
                    elif classes > 0:
                        sample_images_cgan(conf_data, epoch)
        if seq == 0:
            log_file.write("[Epoch %d/%d] [D loss: %f] [G loss: %f] \n" % (epoch, epochs,conf_data['d_loss'][0], conf_data['g_loss']))      
        elif seq == 1:
            # print ("Done")
            log_file.write("[Epoch %d/%d] [D loss: %f] [G loss: %f] \n" % (epoch, epochs,conf_data['d_loss'], conf_data['g_loss']))           
    conf_data['generator_model'] = generator
    conf_data['discriminator_model'] = discriminator
    conf_data['combined_model'] = combined_model
    conf_data['log_file'] = log_file
    return conf_data