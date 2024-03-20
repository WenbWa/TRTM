
import torch
import cv2 as cv
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)

class Renderer:
    def __init__(self, camera_height=2.6, image_size=720, dtype=torch.float32, device=torch.device("cuda")):
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.image_size = image_size
        self.camera_height = camera_height

        # init pytorch3d top-view camera
        self.R, self.T = look_at_view_transform(dist=self.camera_height, elev=torch.tensor([0.0]), azim=torch.tensor([0.0]))
        self.camera = FoVPerspectiveCameras(device=self.device, R=self.R, T=self.T)
        # init rasterize setting for mesh rendering
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * 1e-8,
            faces_per_pixel=1
        )
        # init mesh depth map renderer
        self.mesh_renderer = MeshRasterizer(
            cameras=self.camera,
            raster_settings=self.raster_settings
        )

        self.ground_value = 200.
        self.normal_value = 250.
        self.scale_value = 240.

    # render images from tensor vertices and faces
    def render_mesh_depth_mask_images(self, vertices, faces):
        # get batch_size
        batch_size = vertices.shape[0]
        # construct mesh object
        mesh = Meshes(verts=[vertices[nb] for nb in range(batch_size)], faces=[faces[nb] for nb in range(batch_size)])
        # render and normalize depth image
        depth_image = self.mesh_renderer(mesh).zbuf[..., 0]
        depth_image[depth_image > 0] = self.ground_value - (self.camera_height - depth_image[depth_image > 0]) * self.normal_value
        depth_image[depth_image <= 0] = 255.
        # mask silhouette image
        mask_image = (depth_image != 255.) * 255.
        return depth_image.type(self.dtype), mask_image.type(self.dtype)

    # convert depth image (0, 1) to mesh point clouds with sample ratio
    def sample_points_from_depth_images(self, depth_images, sample_ratio=0.05):
        # get batch size
        batch_size = depth_images.shape[0]
        # sample points within each depth image
        batch_depth_points = []
        for ni in range(batch_size):
            # convert depth images to depth points [index, value]
            depth_index = torch.where(depth_images[ni] < self.ground_value)
            depth_value = depth_images[ni][depth_images[ni] * 255. < self.ground_value].view(-1, 1) * 255.
            depth_points = torch.cat([depth_index[0].unsqueeze(1), depth_index[1].unsqueeze(1), depth_value], dim=1)
            # sample depth points
            depth_points = depth_points[torch.randperm(depth_points.shape[0])[:int(depth_points.shape[0]*sample_ratio)]]
            # centralize and normalize depth points
            temp_points = depth_points.clone()
            depth_points[:, 0] = (temp_points[:, 1] - self.image_size / 2) / self.scale_value
            depth_points[:, 1] = (- temp_points[:, 0] + self.image_size / 2) / self.scale_value
            depth_points[:, 2] = (self.ground_value - temp_points[:, 2]) / self.normal_value
            batch_depth_points.append(depth_points)
        return batch_depth_points

