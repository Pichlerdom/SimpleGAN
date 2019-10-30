import numpy as np
import matplotlib.pyplot as plt

class SimpleMetrics:
    
    def __init__(self,name,config):
        self.name = name
        self.data = []
        self.config = config

    def export_graph(self):
        plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        plt.plot(self.data, '-', color='g')
        plt.legend(loc='best', fontsize=16)
        # plt.ylabel("Average Reward", fontsize=16)
        plt.xlabel("Step", fontsize=16)
        plt.title(self.name, fontsize=20)
        ax.tick_params(labelsize=16)
        plt.savefig(self.config.graph_dir + self.name + ".png")
        print("Graph " + self.name + " exported to" + self.config.graph_dir + self.name + ".png")
        plt.close()
        
    def dump_csv(self):
        path = self.config.metrics_dir + "dump_" + self.name + ".csv" 
        f = open(path,"w")
        for d in self.data:
            for v in d:
                f.write(str(v))
                f.write(" ")
            f.write("\n")
        f.close()

    def update(self):
        pass

class CombinedModelMetrics(SimpleMetrics):
    def __init__(self,name,config):
        super().__init__(name,config)
        self.training_disc = []

    def model_callback(self,model):
        self.data.append(model.combined_model_metrics[1] * 100)
        self.training_disc.append(model.training_disc)
        self.update()

    def dump_csv(self):
        path = self.config.metrics_dir + "dump_" + self.name + ".csv" 
        f = open(path,"w")
        for v in self.data:
            f.write(str(v))
            f.write("\n")
        f.close()

class DiscriminatorMetrics(SimpleMetrics):
    def model_callback(self,model):
        self.data.append(model.discriminator_metrics)
        self.update()
