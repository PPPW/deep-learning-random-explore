from fastai.basic_train import Recorder
from tqdm import trange


class LearnerMock():
    def __init__(self, data_size):
        self.data = DataBunchMock(data_size)
        self.opt = OptimizerMock()
        self.callbacks = [Recorder(self)]
        
    def fit(self, epochs, callbacks):
        rc = Recorder(self)
        cb = callbacks[0]   
        
        rc.on_train_begin(pbar=PBarMock(), metrics_names=[])
        cb.on_train_begin(n_epochs=epochs)
        
        for _ in trange(epochs, desc='Epoch'):
            for _ in range(len(self.data.train_dl)):
                rc.on_batch_begin(True)
                cb.on_batch_end(True)


class DataBunchMock():
    def __init__(self, data_size):
        self.train_dl = [0]*data_size
        

class OptimizerMock():
    def __init__(self):
        self.lr = 0.1
        self.mom = 0.9


class PBarMock():
    def write(self, msg, table):
        pass