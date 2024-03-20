import os, glob
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class ClothDataset(Dataset):
    """
        Datasets: data_path/<real, simu>/<train, val, test>/<*.png, *.txt>
        Sample: {'image_real', 'mesh_real', 'image_simu', 'mesh_simu'}
    """

    def __init__(self, phase=None, data_path=None, data_transform=None, use_real=False, use_simu=False, sample_ratio=1.0):
        # dataset phase: train, val, test
        self.phase = phase
        # dataset path: data_path/<real, simu>/<train, val, test>/<*.png, *.txt>
        self.data_path = data_path
        # dataset transform
        self.data_transform = data_transform

        # dataset size: real and simu images of dragged, folded, and dropped clothes
        self.data_real = sorted(glob.glob(os.path.join(self.data_path, 'real', self.phase, '*.real_depth.png')))
        self.data_simu = sorted(glob.glob(os.path.join(self.data_path, 'simu', self.phase, '*.simu_depth.png')))
        self.data_size_simu = len(self.data_simu)
        self.data_size_real = len(self.data_real)

        # use real data, use simu data
        self.use_real = use_real
        self.use_simu = use_simu
        # dataset sample_ratio: 0.5 / 1.0
        self.sample_ratio = sample_ratio

        # dataset size: use simu data or use real data
        self.data_size = self.data_size_real if self.use_real and not self.use_simu else self.data_size_simu
        print('DataLoader phase: {}; use_simu: {}, simu_size: {}; use_real: {}, real_size: {}'.format(
            self.phase, self.use_simu, self.data_size_simu, self.use_real, self.data_size_real))

    def __len__(self):
        # return sampled data_size
        return int(self.data_size * self.sample_ratio)

    def __getitem__(self, idx):
        # load sampled data_sample
        sample = self.load_sample(min(int(idx * (1 / self.sample_ratio)), self.data_size))
        # apply data transform
        if self.data_transform is not None:
            sample = self.data_transform(sample)
        return sample

    # load cloth sample: {'image_real', 'mesh_real', 'image_simu', 'mesh_simu'}
    def load_sample(self, idx):
        # init empty sample
        sample = dict()
        # train/val/test default: data_path/<real, simu>/<train, val, test>/<*.png, *.txt>
        if self.phase in ['train', 'val', 'test']:
            # load real image for train/val
            if self.use_real:
                image_real_path = self.data_real[idx % self.data_size_real]
                sample['image_name'] = image_real_path.split('/')[-1].split('.')[0]
                sample['image_real_input'] = cv.resize(cv.imread(image_real_path), (720, 720))  # always resize to (720, 720)
                sample['image_real'] = sample['image_real_input'].copy()
            # load simu image and mesh for train/val
            if self.use_simu:
                image_simu_path = self.data_simu[idx % self.data_size_simu]
                mesh_simu_path = image_simu_path.replace('simu_depth.png', 'simu_mesh.txt')
                sample['image_name'] = image_simu_path.split('/')[-1].split('.')[0]
                sample['image_simu_input'] = cv.imread(image_simu_path)
                sample['image_simu_input'] = cv.resize(cv.imread(image_simu_path), (720, 720))  # always resize to (720, 720)
                sample['image_simu'] = sample['image_simu_input'].copy()
                sample['mesh_simu'] = np.loadtxt(mesh_simu_path)
        else:
            raise ValueError('Unknown phase {} from [{}, {}, {}]'.format(self.phase, 'train', 'val', 'test'))
        return sample


