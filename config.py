import sys
import os

class Config:
    show_window = True
    export_graphs = True 
    auto_export_model = False
    keyboard_enable = True
    save_interval = 10
    data_dir = ""
    save_dir = ""
    graph_dir = ""

    DATA_SET = os.getcwd() + "/jaffedbase/jaffe"
    GRAPH_DIR = os.getcwd() + "/graphs/"
    SAVE_DIR = os.getcwd() + "/models/"
    METRICS_DIR = os.getcwd() + "/metrics/"
    CONFIG_TEXT = '\n\n\n -nw \t\t no window '\
        '\n -ng \t\t no graph export '\
        '\n -aem \t\t turn on auto export model '\
        '\n -si <number> \t how many iterations befor the model is saved again'\
        '\n -ds <path> \t specify the data set folder'\
        '\n -lm <name> \t specify a model name to load'\
        '\n\n -?  \t\t print this'
        

    
    def __init__(self,load_name = None):
        self.data_dir = self.DATA_SET
        self.save_dir = self.SAVE_DIR
        self.graph_dir = self.GRAPH_DIR
        self.load = not (load_name == None)
        self.load_name = load_name
        self.metrics_dir = self.METRICS_DIR
        
        self.parsArgs()

    def parsArgs(self):
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if(arg == "-nw"):
                self.show_window = False
            elif(arg == "-nk"):
                self.keyboard_enable = False
            elif(arg == "-ng"):
                self.export_graphs = False
            elif(arg == "-aem"):
                self.auto_export_model = True
            elif(arg == "-?"):
                print(self.CONFIG_TEXT)
                exit();
            elif(arg == "-si"):
                i += 1
                arg = sys.argv[i]
                try:
                    self.save_interval = int(arg)
                except:
                    print("Save interval is not a number!\n\n")
                    print(self.CONFIG_TEXT)
            elif(arg == "-ds"):
                i += 1
                arg = sys.argv[i]
                self.data_dir = arg
                self.DATA_SET = arg
            elif(arg == "-lm"):
                i += 1
                arg = sys.argv[i]
                self.load_name = arg
                self.load = True
            else:
                print("\n\nArgument \"%s\" not supported!"%(arg))
            i += 1
                
