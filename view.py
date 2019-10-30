import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout 
from qt_widgets import CombinedModelMetricsGraph, DiscriminatorMetricsGraph

class MetricsView(QMainWindow):
    def __init__(self,model,config):
        super().__init__()
        
        self.config = config
        
        self.title = "metrics"
        self.width = 640
        self.height = 240
        self.top = 100
        self.left = 100
        self.model = model

        self.init_window()
        
    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top,self.left,self.width,self.height)
        self.window = QWidget()

        vbox_layout = QVBoxLayout()
        self.combined_model_metrics = CombinedModelMetricsGraph(self.window, self.width, self.height/2,"generator",self.config)
        self.discriminator_metrics = DiscriminatorMetricsGraph(self.window, self.width, self.height/2,"discriminator",self.config) 
        self.model.register_data_logger(self.combined_model_metrics)
        self.model.register_data_logger(self.discriminator_metrics)
        
        vbox_layout.addWidget(self.combined_model_metrics)
        vbox_layout.addWidget(self.discriminator_metrics)
        
        self.window.setLayout(vbox_layout)
        self.setCentralWidget(self.window)
        self.show()

    def export_graphs(self):
        self.discriminator_metrics.export_graph()
        self.combined_model_metrics.export_graph()        

    def dump_csv(self):
        self.discriminator_metrics.dump_csv()
        self.combined_model_metrics.dump_csv()
