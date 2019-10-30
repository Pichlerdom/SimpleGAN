import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from numpy.random import random,choice

class TrainingDataGen:

    def __init__(self,image_dir):
        self.init_image_loader()
        self.image_dir = image_dir
        
    def init_image_loader(self):
        data_gen_args = dict(featurewise_center=False,
                             rotation_range=1,
                             width_shift_range=1.0,
                             height_shift_range=1.0,
                             horizontal_flip=True,
                             vertical_flip=False)
        
        self.image_loader = ImageDataGenerator(**data_gen_args)
        
    def get_fake_images(self,generator,num,batch_size):
        images = []
        for i in range(num):
            latent_space = np.random.normal(0,1,(batch_size,64))
            images.append(generator.predict(latent_space,batch_size = batch_size))
        return images

    def get_real_images(self,num,w,h):
        batch = self.image_loader.flow_from_directory(self.image_dir,target_size=(h,w),batch_size = num,color_mode='grayscale',class_mode = 'binary',shuffle = True)
        return self.normalize_images(batch[0][0])

    def normalize_images(self,images):
        images = images.astype('float32')
        images = (images -127.5)/127.5
        return images
    
    def denormalize_images(self,images):
        images = (images * 127.5) + 127.5;
        return images

    def smooth_positive_labels(self,y):
        return y - 0.3 + (random(y.shape) * 0.5)

    def smooth_negativ_labels(self,y):
        return y + (random(y.shape) * 0.3)

    def noisy_labels(self,y, p_flip):
        # determine the number of labels to flip
        n_select = int(p_flip * y.shape[0])
        # choose labels to flip
        flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
        # invert the labels in place
        y[flip_ix] = 1 - y[flip_ix] 
        return y

    def get_fake_labels(self,num):
        labels = np.zeros(num)
        labels = self.smooth_negativ_labels(labels)
        return self.noisy_labels(labels,0.05)
    
    def get_real_labels(self,num):
        labels = np.ones(num)
        labels = self.smooth_positive_labels(labels)
        return self.noisy_labels(labels,0.05)
