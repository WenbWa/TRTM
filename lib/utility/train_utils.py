import numpy as np
import sys, torch, random, logging

# ------------------- Train Utility ------------------- #

class AverageMeter(object):
    """
    Average agent
    """
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# set system seeds
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# set seed with worker_id
def worker_init(worker_id, main_seed):
    seed = worker_id + main_seed
    set_seed(seed)

# save train logger
def add_logging(name, file, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0: return logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# convert tensor to numpy
def torch2numpy(tensor):
    return tensor.detach().cpu().numpy()

# convert numpy to torch
def numpy2torch(np_array):
    return torch.from_numpy(np_array).float()

