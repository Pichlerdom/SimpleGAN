import os
import sys
import numpy as np
from threading import Thread 
from keras_model_gan import KerasGAN
from view import MetricsView
from metrics import CombinedModelMetrics, DiscriminatorMetrics
from PyQt5.QtWidgets import QApplication
import time    
from config import Config
from utils import set_tf_loglevel
import logging

set_tf_loglevel(logging.FATAL)

config = Config()
app = None 
run = True
save_flag = False

print("Training set dir: " + config.data_dir)

gan_model = KerasGAN(config)

def keyboard_handle():
    global save_flag
    global run
    global app
    while(run):
        command = sys.stdin.read(1);
        if(command == 's'):
            print("Saving!")
            save_flag = True
        elif(command == 'q'):
            print("Ending!")
            run = False
            if(app != None):
                app.quit()
        else:
            time.sleep(0.1)
            
def training_thread_foo():
    global gan_model
    global run
    time.sleep(1)
    while(run):
        gan_model.train_one_step()
    
if(config.keyboard_enable):    
    keyboard_thread = Thread(target = keyboard_handle)
    keyboard_thread.start()
    
    
if(config.show_window):
    training_thread = Thread(target = training_thread_foo)
    training_thread .start()
    
    app = QApplication(sys.argv)
    metrics_view = MetricsView(gan_model,config)
    app.exec()

    training_thread.join()
else:    
    discriminator_metrics = DiscriminatorMetrics("discriminator",config)
    combined_model_metrics = CombinedModelMetrics("combined_model",config) 
    gan_model.register_data_logger(discriminator_metrics)
    gan_model.register_data_logger(combined_model_metrics)
    training_thread_foo()
    discriminator_metrics.dump_csv()
    combined_model_metrics.dump_csv()
    
run = False
if(config.keyboard_enable):
    keyboard_thread.join()
    
gan_model.save_gan("end_save")

if(config.show_window):
    metrics_view.export_graphs()
    metrics_view.dump_csv()

print("end")

