# # Test Cloth-GNN Model on template_square, use_real only, from checkpoints/args.checkpoint
# python main.py --phase test --cloth_name template_square --use_real True
# # Train Cloth-GNN Model on template_square, use_simu only, save to checkpoints/exp_train
# python main.py --phase train --cloth_name template_square --use_simu True

import torch
import numpy as np
import os, sys, pickle, argparse

from lib.utility.train_utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from lib.trainer import Trainer
from lib.model.cloth_model import ClothModel
from lib.data.cloth_dataset import ClothDataset
from lib.data.cloth_transform import NoiseAugmentation, RotateAugmentation, AssignMeshIndex, ReshapeImage, NumpyToTensor


PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, 'datasets')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')


## ------------------- Set Seed and Threads ------------------- ##
main_seed = 0
set_seed(main_seed)
num_threads = 8

## ------------------- Set System Parser ------------------- ##
parser = argparse.ArgumentParser()
parser.add_argument("--phase", help="train/test", type=str, default="test")
parser.add_argument("--cloth_name", help="template_square", type=str, default="template_square")
parser.add_argument("--use_real", type=bool, default=False)
parser.add_argument("--use_simu", type=bool, default=False)
parser.add_argument("--sample_ratio", type=float, default=0.5)  # train with half dataset to save time
parser.add_argument("--save_name", help="exp_name", type=str, default="exp_train_half")  # save train checkpoint
parser.add_argument("--checkpoint", type=str, default='checkpoint_large_size.pt')  # load test checkpoint
parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
parser.add_argument("--epoch_size", help="epoch_size", type=int, default=200)
parser.add_argument("--augment_size", help="augment_size", type=int, default=8)
parser.add_argument("--image_size", help="image_size", type=int, default=720)
parser.add_argument("--lr", help="learning rate of optimizer", type=float, default=1e-4)
parser.add_argument("--save_step", type=int, default=50)
parser.add_argument("--schedule_step", type=int, default=100)
args = parser.parse_args()
print('Phase: {}, Cloth: {}, Use_Real: {}, Use_Simu: {}'.format(args.phase, args.cloth_name, args.use_real, args.use_simu))

# load template info
template_info = pickle.load(open(os.path.join(DATASET_DIR, 'template_square.pickle'), mode='rb'))
print('Template_Info:', [k for k, v in template_info.items()])
# set device
device = torch.device('cuda')
# init Cloth Model
model = ClothModel(template_info).to(device)
# init Adam optimizer
opt = torch.optim.Adam(model.parameters(), lr=args.lr)


# set up trainer
trainer = Trainer(
    model,
    opt,
    template_info,
    CHECKPOINT_DIR,
    os.path.join(DATASET_DIR, args.cloth_name),
    args.batch_size,
    args.epoch_size,
    args.image_size,
    args.lr,
    args.save_step,
    args.schedule_step,
)

# train cloth_model with data_loader_train and data_loader_val
if args.phase == "train":
    # init data transform for train
    data_transform_train = Compose([NoiseAugmentation(noise_range=2),
                                    RotateAugmentation(random_rotate=90, uniform_rotate=1),
                                    AssignMeshIndex(template_info=template_info),
                                    ReshapeImage(image_size=(224, 224)),
                                    NumpyToTensor()])
    # init data loader for train
    data_loader_train = DataLoader(
        dataset=ClothDataset(phase='train',
                             data_path=os.path.join(DATASET_DIR, args.cloth_name),
                             data_transform=data_transform_train,
                             use_real=args.use_real,
                             use_simu=args.use_simu,
                             sample_ratio=args.sample_ratio),
        batch_size=args.batch_size,
        shuffle=True,  # re-shuffle data at every epoch
        num_workers=num_threads,  # number of worker threads batching data
        drop_last=True,  # if last batch not of size batch_size, drop
        pin_memory=True,  # faster data transfer to GPU
        worker_init_fn=lambda x: worker_init(x, main_seed),  # seed all workers. Important for reproducibility
    )

    # init data transform for val
    data_transform_val = Compose([NoiseAugmentation(noise_range=2),
                                  RotateAugmentation(random_rotate=90, uniform_rotate=1),
                                  AssignMeshIndex(template_info=template_info),
                                  ReshapeImage(image_size=(224, 224)),
                                  NumpyToTensor()])
    # init data loader for val
    data_loader_val = DataLoader(
        dataset=ClothDataset(phase='val',
                             data_path=os.path.join(DATASET_DIR, args.cloth_name),
                             data_transform=data_transform_val,
                             use_real=args.use_real,
                             use_simu=args.use_simu,
                             sample_ratio=args.sample_ratio),
        batch_size=args.batch_size,
        shuffle=False,  # go through the test data sequentially. Easier to plot same samples to observe them over time
        num_workers=num_threads,  # number of worker threads batching data
        drop_last=False,  # we want to validate on ALL data
        pin_memory=True,  # faster data transfer to GPU
        worker_init_fn=lambda x: worker_init(x, main_seed),
    )
    # locate save experiment dir
    save_dir = os.path.join(CHECKPOINT_DIR, args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    # train model with dataloader train and val
    trainer.train_model(save_dir, data_loader_train, data_loader_val)

# test cloth_model with data_loader_test
elif args.phase == "test":
    # init data transform for test
    data_transform_test = Compose([NoiseAugmentation(noise_range=2),
                                   RotateAugmentation(random_rotate=0, uniform_rotate=args.augment_size),
                                   AssignMeshIndex(template_info=template_info),
                                   ReshapeImage(image_size=(224, 224)),
                                   NumpyToTensor()])
    # init data loader for test
    data_loader_test = DataLoader(
        dataset=ClothDataset(phase='test',
                             data_path=os.path.join(DATASET_DIR, args.cloth_name),
                             data_transform=data_transform_test,
                             use_real=args.use_real,
                             use_simu=args.use_simu,
                             sample_ratio=1.0),
        batch_size=args.batch_size,
        shuffle=False,  # go through the test data sequentially
        num_workers=num_threads,  # number of worker threads batching data
        drop_last=False,  # we want to test on ALL data
        pin_memory=True,  # faster data transfer to GPU
        worker_init_fn=lambda x: worker_init(x, main_seed),
    )
    # locate checkpoint
    checkpoint_fn = os.path.join(CHECKPOINT_DIR, args.checkpoint)
    assert os.path.isfile(checkpoint_fn)
    # test model with dataloader test
    trainer.test_model(checkpoint_fn, data_loader_test, args.augment_size)
