import torch
import numpy as np
import cv2 as cv
from lib.utility.cloth_utils import *


class NoiseAugmentation:
    """
        Add Gaussian Noise to image
    """
    def __init__(self, noise_range):
        self.noise_range = noise_range

    def __call__(self, sample):
        # noise augmentation
        sample = self.add_gaussian_image_noise(sample, self.noise_range)
        return sample

    # add gaussian noise to image_simu
    def add_gaussian_image_noise(self, sample, noise_range):
        # return sample if zero range
        if noise_range == 0: return sample
        # loop over all sample items
        for k, v in sample.items():
            # noise image_simu with low and high frequency gaussian noise
            if k in ['image_simu']:
                # generate low and high frequency gaussian noise
                low_fre_noise = gaussian_noise(sample[k], mu=-2, sigma=3, fre=2)
                high_fre_noise = gaussian_noise(sample[k], mu=-2, sigma=9, fre=36)
                # add gaussian noise to image
                sample[k + '_input'] = noise_image(sample[k + '_input'], (noise_range > 0)*low_fre_noise + (noise_range > 1)*high_fre_noise)
                sample[k] = noise_image(sample[k], (noise_range > 0)*low_fre_noise + (noise_range > 1)*high_fre_noise)
        return sample


class RotateAugmentation:
    """
        Random or Uniform Rotate image, mesh, mesh_index for train/val/test
    """
    def __init__(self, random_rotate=90, uniform_rotate=1):
        self.random_rotate = random_rotate
        self.uniform_rotate = uniform_rotate

    def __call__(self, sample):
        # random rotation augmentation
        if self.random_rotate == 90 or self.random_rotate == 360:
            # get random angle within [0, 90, 180, 270]
            rotate_angle = np.random.choice([90, 180, 270])
            # rotate image, mesh, mesh_index_order
            if np.random.uniform() <= 0.5:
                sample = self.random_rotate_sample(sample, rotate_angle)
            # rotate augmentation within 360 degrees
            if self.random_rotate == 360:
                # get random angle within (-45, 45)
                rotate_angle = np.random.uniform(0, 1) * 90 - 45
                # rotate image, mesh, mesh_index_order
                if np.random.uniform() <= 0.5:
                    sample = self.random_rotate_sample(sample, rotate_angle)

        # uniform rotation augmentation
        sample = self.uniform_rotate_sample(sample, self.uniform_rotate)
        return sample

    # random rotate sample images and meshes with angle
    def random_rotate_sample(self, sample, angle):
        # loop over all sample items
        for k, v in sample.items():
            # rotate image with angle
            if k in ['image_simu', 'image_real']:
                sample[k + '_input'] = rotate_image(sample[k + '_input'], angle)
                sample[k] = rotate_image(sample[k], angle)
            # rotate mesh with angle
            elif k in ['mesh_simu', 'mesh_real']:
                sample[k] = rotate_mesh(sample[k], angle)
        return sample

    # uniform rotate sample images and meshes with time
    def uniform_rotate_sample(self, sample, time):
        # loop over all sample items
        for k, v in sample.items():
            # rotate image with time
            if k in ['image_simu', 'image_real']:
                sample[k + '_input'] = np.asarray([rotate_image(sample[k + '_input'], nt * (360 // time)) for nt in range(time)])
                sample[k] = np.asarray([rotate_image(sample[k], nt * (360 // time)) for nt in range(time)])
            # rotate mesh with time
            elif k in ['mesh_simu', 'mesh_real']:
                sample[k] = np.asarray([rotate_mesh(sample[k], nt * (360 // time)) for nt in range(time)])
        return sample


class AssignMeshIndex:
    """
        Assign mesh index according to template index
    """
    def __init__(self, template_info=None):
        self.template_mesh = template_info['mesh_pos']

    def __call__(self, sample):
        # assign mesh index
        sample = self.assign_mesh_index_as_template(sample)
        return sample

    # assign mesh index according to template
    def assign_mesh_index_as_template(self, sample):
        # loop over all sample items
        for k, v in sample.items():
            if k in ['mesh_simu', 'mesh_real']:
                # assign mesh index for each augmented mesh
                assign_meshes = [assign_mesh_index(sample[k][na], self.template_mesh) for na in range(sample[k].shape[0])]
                sample[k] = np.asarray(assign_meshes)
        return sample


class ReshapeImage:
    """
        Resize sample image to (224, 224), normalize to (0, 1)
    """
    def __init__(self, image_size=(224, 224)):
        self.image_size = tuple(image_size)

    def __call__(self, sample):
        # loop over all sample items
        for k, v in sample.items():
            if k in ['image_simu', 'image_real']:
                # resize image to (224, 224), convert from (A, H, W, C) to (A, C, H, W), normalize from (0, 255) to (0, 1)
                sample[k] = np.asarray([(cv.resize(sample[k][na], self.image_size)).transpose(2, 0, 1) / 255. for na in range(sample[k].shape[0])])
        return sample

class NumpyToTensor:
    """
        Convert numpy sample to torch tensor
    """
    def __call__(self, sample):
        # loop over all sample items
        for k, v in sample.items():
            # convert numpy to torch tensor
            if k == 'image_name': continue
            sample[k] = torch.from_numpy(sample[k]).float()
        return sample
