import numpy as np
import torch
from setting import params

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.nseqs = src.size(0)
        self.ntokens = (self.src != pad_index).data.sum().item()
        
        # self.trg = None

        # if trg is not None:
        self.trg = trg - 1
        
        if params["USE_CUDA"]:
            self.src = self.src.cuda()

            # if trg is not None:
            self.trg = self.trg.cuda()



def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.text, batch.label, pad_idx)





