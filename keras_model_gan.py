
import numpy as np
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Reshape,Activation,BatchNormalization,LeakyReLU
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Conv2DTranspose
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from cv2 import imwrite
from training_data_gen import TrainingDataGen

class KerasGAN:

    generator_batch_size = 16
    discriminator_batch_size = 32
    combined_model_batch_size = 16    

    generator_batches = 4
    discriminator_batches = 4
    combined_model_batches = 4
    
    def __init__(self,config):
        self.config = config
        self.data_gen = TrainingDataGen(config.data_dir)

        self.kernel_init = RandomNormal(mean=0.0, stddev=0.02) 

        self.learning_rate = 0.001 #has no effect because of keras tensorflow_v2
        self.loss = 'binary_crossentropy'
        
        if(config.load):
            self.load_gan(config.load_name)
        else:
            self.image_height = 8
            self.image_width = 8
            self.latent_space_size = self.image_height * self.image_width

            self.init_discriminator()
            self.init_generator()
            self.init_combined_model()
            
            self.current_network_depth = 1
            self.iteration = 0
            
        self.data_loggers = []
        
    def init_discriminator(self):
        layers = [
            Input(shape=(self.image_height,self.image_width,1,)),
            Conv2D(16,(4,4),kernel_initializer = self.kernel_init,use_bias = False),
            LeakyReLU(),
            BatchNormalization(),
            Flatten(),
            Dense(self.image_height * self.image_width,activation = 'relu',kernel_initializer = self.kernel_init),
            Dropout(0.5),
            Dense(1,activation = 'sigmoid')
        ]

        self.discriminator = self.make_model(layers)
        self.discriminator = self.compile_model(self.discriminator)

        self.discriminator_first_layer_to_unlock = 0
        self.discriminator_last_layer_to_unlock = len(layers) #-1 ?
        
        
    def init_generator(self):
        layers = [
            Dense(self.latent_space_size,input_dim = self.latent_space_size,activation='tanh'),
            Reshape((self.image_height,self.image_width,1)),
            BatchNormalization(),
            Conv2DTranspose(64,(4,4),strides=(1,1),kernel_initializer=self.kernel_init,padding='same',use_bias = False),
            LeakyReLU(),
            BatchNormalization(),
            Conv2DTranspose(1,(4,4),strides=(1,1),kernel_initializer=self.kernel_init,padding='same',use_bias = False),
        ]
        
        self.generator = self.make_model(layers)
        self.generator = self.compile_model(self.generator)

        self.generator_first_layer_to_unlock = 0
        self.generator_last_layer_to_unlock = len(layers) #-1 ?
        
    def init_combined_model(self):
        self.combined_model = self.make_model(self.generator.layers + self.lock_layers(self.discriminator.layers))
        self.combined_model = self.compile_model(self.combined_model)

    def increase_discriminator_depth(self):
        pass
    
    def increase_generator_depth(self):
        pass

    def train_one_step(self):
        metrics = self.test_discriminator()        

        if(metrics[0] < self.disc_min_hit_rate_real or metrics[1] < self.disc_min_hit_rate_fake):
            self.train_discriminator()
        else:
            self.train_combined_model()
            self.test_combined_model()
            
        self.update_data_logger()
        self.iteration += 1
        if(self.iteration%self.config.save_interval == 0):
            self.save_gan("auto_save_" + str(self.iteration))
            
    def train_combined_model(self):
        generator_x = np.random.normal(0,1,(self.combined_model_batch_size * self.combined_model_batches,self.latent_space_size))
        generator_y = np.ones(self.combined_model_batch_size * self.combined_model_batches)
        
        self.combined_model.fit(generator_x,generator_y, batch_size=self.combined_model_batch_size,
                                epochs = self.combined_model_batches)
        
    def test_combined_model(self):#model,BATCH_SIZE_GEN):
        accuracy = 0;
        x = np.random.normal(0,1,(self.combined_model_batch_size * self.combined_model_batches,self.latent_space_size))
        y = np.ones(self.combined_model_batch_size * self.combined_model_batches)
        
        metrics = self.combined_model.evaluate(x, y,batch_size = self.combined_model_batch_size)        
        print("gena:%f%%"%((metrics[1])*100))

    def train_discriminator(self):
        self.enable_model_learning(self.discriminator)
    
        fake_x = self.data_gen.get_fake_images(self.generator,self.discriminator_batch_size * self.discriminator_batches)
        fake_y = self.data_gen.get_fake_labels(self.discriminator_batch_size * self.discriminator_batches)
                
        self.discriminator.fit(fake_x,fake_y,batch_size=self.discriminator_batch_size)
        
        real_x = self.data_gen.get_real_images(self.discriminator_batch_size * self.discriminator_batches,
                                               self.image_width,self.image_height)
        real_y = self.data_gen.get_real_labels(self.discriminator_batch_size * self.discriminator_batches)

        self.discriminator.fit(real_x,real_y,batch_size=self.discriminator_batch_size)

        self.lock_model_learning(self.discriminator)

    def test_discriminator(self):    
        fake_x = self.data_gen.get_fake_images(self.generator,
                                               self.discriminator_batch_size * self.discriminator_batches)
        fake_y = np.zeros(self.discriminator_batch_size * self.discriminator_batches)
        
        real_x = self.data_gen.get_real_images(self.discriminator_batch_size * self.discriminator_batches,
                                               self.image_width,self.image_height)
        real_y = np.ones(self.discriminator_batch_size * self.discriminator_batches)
        
        metrics_fake = self.discriminator.evaluate(fake_x,fake_y,batch_size = self.discriminator_batch_size)
        metrics_real = self.discriminator.evaluate(real_x,real_y,batch_size = self.discriminator_batch_size)    
        metrics = (metrics_real[1] + metrics_fake[1])/2

        print ("disc     :%f%%"%((metrics) * 100))
        print ("disc_fake:%f%%"%((metrics_fake[1]) * 100))
        print ("disc_real:%f%%"%((metrics_real[1]) * 100))
        
        image = self.data_gen.denormalize_images(self.generator.predict_on_batch(np.zeros((1,self.latent_space_size))))[0]

        imwrite('/home/dominik/Development/Desktop_Linz/output/%d.png'%self.iteration,np.array(image))
        return np.array([metrics * 100,metrics_fake[1] * 100,metrics_real[1] * 100])

    def make_model(self,layers):
        model = Sequential()
        for l in layers:
            print(l)
            print()
            model.add(l)
    
        return model
    
    def compile_model(self,model):
        #opt = Adam(lr = self.learning_rate,beta_1=0.5)#SGD(lr = learning_rate, decay = 1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=self.loss, optimizer="adam", metrics = ['acc'])
        return model;

    def lock_layers(self,layers):
        for l in layers:
            l.trainable = False
        return layers

    def register_data_logger(self,data_logger):
        self.data_loggers.append(data_logger)

    def update_data_logger(self):
        for logger in self.data_loggers:
            logger.keras_model_event(self)
            
    def save_gan(self,name):
        path = config.save_dir + name + "/"
        self.save_model(path +"gen",self.generator)
        self.save_model(path + "disc",self.discriminator)

        f = open(path + "metadata.txt", "w")
        f.write(self.discriminator_first_layer_to_unlock + ",")
        f.write(self.discriminator_last_layer_to_unlock + ",")
        f.write(self.generator_first_layer_to_unlock + ",")
        f.write(self.generator_last_layer_to_unlock + ",")
        f.write(self.image_width + ",")
        f.write(self.image_height + ",")
        f.write(self.iteration)
        f.close()

    def load_gan(self,name):
        path = config.save_dir + name + "/"
        self.generator =  self.load_model(path + "gen")
        self.discriminator =  self.load_model(path + "disc")
        self.combined_model = self.make_model(self.generator.layers + self.lock_layers(self.discriminator.layers))
        self.generator = self.compile_model(self.generator)
        self.discriminator = self.compile_model(self.discriminator)

        f = open(path + "metadata.txt" , "r")
        self.discriminator_first_layer_to_unlock = self.get_next_value(f)
        self.discriminator_last_layer_to_unlock = self.get_next_value(f)
        self.generator_first_layer_to_unlock = self.get_next_value(f)
        self.generator_last_layer_to_unlock = self.get_next_value(f)
        self.image_width = self.get_next_value(f)
        self.image_height = self.get_next_value(f)
        self.iteration = self.get_next_value(f)
        f.close()

    def get_next_value(self,f):
        value = ""
        while(True):
            c = f.read(1)
            if(c == ',') or (not c):
                return int(value)
            else:
                value += str(c)
            
    def load_model(self,name):
        # load json and create model
        json_file = open(name+'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(name+"model.h5")
        print("Loaded model from disk")
        return loaded_model

    def save_model(self,name,model):
        # serialize model to JSON
        model_json = model.to_json()
        with open(name+"model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(name+"model.h5")
            print("Saved model to disk")
'''
    def lock_model_learning_from_to(self,model,from_index, to_index):
        if to_index < from_index:
            return
        for i in range(to_index - from_index):
            model.layers[i + from_index].trainable = False

    def enable_model_learning_from_to(self,model,from_index, to_index):
        if to_index < from_index:
            return
        for i in range(to_index - from_index):
            model.layers[i + from_index].trainable = True
                        
    def lock_model_learning(self,model):
        for l in model.layers:
            l.trainable = False
            
    def enable_model_learning(self,model):
        for l in model.layers:
            l.trainable = True
'''
