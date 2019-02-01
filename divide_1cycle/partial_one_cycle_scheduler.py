from fastai.basic_train import Learner
from fastai.callbacks.one_cycle import OneCycleScheduler
from fastai.core import Floats,Any


class PartialOneCycleScheduler(OneCycleScheduler):
    def __init__(self, learn:Learner, lr_max:float,                  
                 moms:Floats=(0.95,0.85), 
                 div_factor:float=25., pct_start:float=0.3,
                 tot_epochs:int=-1, start_epoch:int=0):
        super().__init__(learn, lr_max, moms, div_factor, pct_start)
        self.start_epoch = start_epoch        
        self.tot_epochs = tot_epochs
            
    def on_train_begin(self, n_epochs:int, **kwargs:Any)->None:
        if self.tot_epochs < 0: self.tot_epochs = n_epochs + self.start_epoch        
        super().on_train_begin(self.tot_epochs, **kwargs)
                      
        self.start_iter = len(self.learn.data.train_dl) * self.start_epoch                
        for _ in range(self.start_iter):
            super().on_batch_end(True) 
