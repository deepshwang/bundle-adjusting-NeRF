import numpy as np
import os, sys, time
import torch
import importlib

import options
from util import log
import wandb

def main():
    # Initialize log
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    # Initialize argument variable
    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):
        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)
        m.load_dataset(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.restore_checkpoint(opt)
        m.initialize_wandb(opt)

        m.train(opt)

if __name__=="__main__":
    main()
