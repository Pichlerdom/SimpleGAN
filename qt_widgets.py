import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout 
from PyQt5.QtGui import QColor, QPen, QPainter
from metrics import SimpleMetrics
import matplotlib.pyplot as plt
        
class SimpleMetricsGraph(SimpleMetrics,QWidget):
    MAX_DATA_POINTS_TO_DISPlAY = 1000

    def __init__(self,parent,width,height,name,config):
        SimpleMetrics.__init__(self,name,config)
        QWidget.__init__(self,parent)
        
        self.width = width
        self.height = height
    
        self.painter = QPainter(self)
        p = self.palette()
        p.setColor(self.backgroundRole(),QColor(0,0,0))
        self.setPalette(p)
        self.setAutoFillBackground(True)

    def resizeEvent(self,event):
        self.height = event.size().height()
        self.width = event.size().width()

    def update(self):
        QWidget.update(self)
        
class CombinedModelMetricsGraph(SimpleMetricsGraph):
    def __init__(self,parent,width,height,name,config):
        super().__init__(parent,width,height,name,config)
        self.training_disc = []

    def model_callback(self,model):
        self.data.append(model.combined_model_metrics[1] * 100)
        self.training_disc.append(model.training_disc)
        self.update()
        
    def paintEvent(self,event):
        offset = 0
        if(len(self.data) > 1):
            lines_per_point = (self.width-10) / (len(self.data) - 1)
            if(len(self.data) >= self.MAX_DATA_POINTS_TO_DISPlAY):
                lines_per_point = (self.width-10) / self.MAX_DATA_POINTS_TO_DISPlAY
                offset = len(self.data) - self.MAX_DATA_POINTS_TO_DISPlAY
        else:
            lines_per_point = (self.width-10)
            return

        self.painter.begin(self)
        height = self.height - 20
        
        for i in range(len(self.data) - 1 - offset):
            v1 = self.data[i + offset]
            v2 = self.data[i + 1 + offset]
            if(self.training_disc[i + 1 + offset]):
                self.painter.setPen(QPen(QColor(155,155,155)))
            else:
                self.painter.setPen(QPen(QColor(0,200,0)))
                
            self.painter.drawLine(int(i * lines_per_point), height - abs(v1/100) * height + 10,
                                  int((i + 1) * lines_per_point),height - abs(v2/100) * height + 10)

            if((i + offset) % 100 == 0):
                self.painter.setPen(QPen(QColor(255,255,255)))
                self.painter.drawLine(int(i * lines_per_point), height, i * lines_per_point, 0)
                self.painter.drawText(int(i * lines_per_point + 10), 10,str(i+offset))
                        
            
        self.painter.setPen(QPen(QColor(255,255,255)))
        self.painter.drawText(10,10,self.name)
        self.painter.end()

    def dump_csv(self):
        path = self.config.metrics_dir + "dump_" + self.name + ".csv" 
        f = open(path,"w")
        for v in self.data:
            f.write(str(v))
            f.write("\n")
        f.close()

class DiscriminatorMetricsGraph(SimpleMetricsGraph):
    COLORS = [QColor(255,0,0), QColor(0,255,0),QColor(0,0,255)]
    NAMES = ["mean", "fake" ,"real"]
    iteration = 0
    def __init__(self,parent,width,height,name,config):
        super().__init__(parent,width,height,name,config)        

    def model_callback(self,model):
        self.data.append(model.discriminator_metrics)
        self.iteration = model.iteration
        self.update()

    def paintEvent(self,event):
        offset = 0
        if(len(self.data) > 1):
            lines_per_point = (self.width-10) / (len(self.data)-1)
            if(len(self.data) >= self.MAX_DATA_POINTS_TO_DISPlAY):
                lines_per_point = (self.width-10) / self.MAX_DATA_POINTS_TO_DISPlAY
                offset = len(self.data) - self.MAX_DATA_POINTS_TO_DISPlAY
        else:
            lines_per_point = (self.width-10)
            return

        lines_per_point = lines_per_point
        self.painter.begin(self)
        
        height = self.height - 20        

        for i in range(len(self.data) - 1 - offset):
            v1 = self.data[i + offset]
            v2 = self.data[i + 1 + offset]
            for j in range(len(v1)):
                self.painter.setPen(QPen(self.COLORS[j]))
                self.painter.drawLine(int(i * lines_per_point), height - abs(v1[j]/100) * height + 10,
                                      int((i + 1) * lines_per_point),height - abs(v2[j]/100) * height + 10)

            if((i + offset) % 100 == 0):
                self.painter.setPen(QPen(QColor(255,255,255)))
                self.painter.drawLine(int(i * lines_per_point), height, i * lines_per_point, 0)
                self.painter.drawText(int(i * lines_per_point + 10), 10,str(i+offset))
                
        self.painter.setPen(QPen(QColor(255,255,255)))
        self.painter.drawText(10,10,self.name + "  epoch: " + str(self.iteration))
        self.painter.end()

    def export_graph(self):
        data =[[],[],[]]
        data[0] = [v[0] for v in self.data]
        data[1] = [v[1] for v in self.data]
        data[2] = [v[2] for v in self.data]
        
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        plt.plot(data[0], '-', color='r',label = "mean")
        plt.plot(data[1], '-', color='g',label = "fake")
        plt.plot(data[2], '-', color='b',label = "real")
        plt.legend(loc='best', fontsize=16)
        # plt.ylabel("Average Reward", fontsize=16)
        plt.xlabel("Step", fontsize=16)
        plt.title(self.name, fontsize=20)
        ax.tick_params(labelsize=16)
        plt.savefig(self.config.graph_dir + self.name + ".png")
        print("Graph " + self.name + " exported to" + self.config.graph_dir + self.name + ".png")
        plt.close()
