import torch
import torch.backends.cudnn as cudnn
from config import *
from dataset import Rooms

if __name__ == '__main__':
    config = Config()

    cudnn.benchmark = True
    torch.cuda.manual_seed_all(config.seed)

    data = Rooms()