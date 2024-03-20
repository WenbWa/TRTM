import os, copy
import torch.nn
import torch.nn.functional as F

from tqdm import tqdm
from lib.utility.cloth_utils import *
from lib.utility.train_utils import *
from lib.utility.render_utils import Renderer
from pytorch3d.loss import chamfer_distance

class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 template_info,
                 exp_dir,
                 data_dir,
                 batch_size,
                 epoch_size,
                 image_size,
                 lr,
                 save_step,
                 schedule_step,
                 dtype=torch.float32,
                 device=torch.device('cuda'),
                 ):

        self.model = model
        self.opt = optimizer
        self.template_info = template_info

        self.exp_dir = exp_dir
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.image_size = image_size
        self.lr = lr
        self.save_step = save_step
        self.schedule_step = schedule_step

        self.dtype = dtype
        self.device = device

        self.L1Loss = torch.nn.L1Loss().to(self.device)
        self.renderer = Renderer(image_size=self.image_size, dtype=dtype, device=device)

        self.log_train_losses = AverageMeter()

    # vertex loss between pred_mesh and true_mesh
    def vertex_loss(self, pred_mesh, true_mesh):
        # calculate batch vertex losses
        batch_vertex_losses = [self.L1Loss(pred_mesh[nb], true_mesh[nb]) for nb in range(true_mesh.shape[0])]
        return batch_vertex_losses

    # corner loss between pred_mesh and true_mesh
    def corner_loss(self, pred_mesh, true_mesh):
        # get corner idx from template
        corner_idx = self.template_info['corner_idx']
        # calculate batch corner losses
        batch_corner_losses = [self.L1Loss(pred_mesh[nb, corner_idx], true_mesh[nb, corner_idx]) for nb in range(true_mesh.shape[0])]
        return batch_corner_losses

    # keypoint loss between pred_mesh and true_mesh
    def keypoint_loss(self, pred_mesh, true_mesh):
        # get keypoint idx from template
        keypoint_idx = self.template_info['keypoint_idx']
        # calculate batch keypoint losses
        batch_keypoint_losses = [self.L1Loss(pred_mesh[nb, keypoint_idx], true_mesh[nb, keypoint_idx]) for nb in range(true_mesh.shape[0])]
        return batch_keypoint_losses

    # vertex wise losses between pred vertices and true vertices
    def vertex_wise_losses(self, pred_mesh, true_mesh, loss_dict):
        # assert pred and true batch size
        assert pred_mesh.shape[0] == true_mesh.shape[0]
        # calculate vertex_loss and keypoint loss
        batch_vertex_losses = torch.stack(self.vertex_loss(pred_mesh, true_mesh))
        batch_keypoint_losses = torch.stack(self.keypoint_loss(pred_mesh, true_mesh))
        # update loss_dict with mean batch_losses
        if 'vertex_loss' in loss_dict: loss_dict['vertex_loss'] += torch.mean(batch_vertex_losses)
        else: loss_dict['vertex_loss'] = torch.mean(batch_vertex_losses)
        if 'keypoint_loss' in loss_dict: loss_dict['keypoint_loss'] += torch.mean(batch_keypoint_losses)
        else: loss_dict['keypoint_loss'] = torch.mean(batch_keypoint_losses)
        # return batch_losses
        return {'batch_vertex_losses': batch_vertex_losses, 'batch_keypoint_losses': batch_keypoint_losses}

    # pixel wise losses between pred vertices and true images
    def pixel_wise_losses(self, pred_vertices, true_images, loss_dict):
        # get batch_size and mesh faces
        batch_size = pred_vertices.shape[0]
        mesh_faces = to_tensor(self.template_info['face_idx'], dtype=torch.long).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        # get true depth and silhouette images
        true_depth = true_images[..., -1]
        true_mask = (true_depth < 255.) * 255.
        # render pred depth and silhouette images
        pred_depth, pred_mask = self.renderer.render_mesh_depth_mask_images(pred_vertices, mesh_faces)
        # assert pred and true batch size
        assert pred_depth.shape[0] == true_depth.shape[0] and pred_mask.shape[0] == pred_mask.shape[0]
        # show_image(to_numpy(true_depth[0].cpu(), dtype=np.uint8))
        # show_image(to_numpy(pred_depth[0].cpu(), dtype=np.uint8))

        # get batch_depth_losses and batch_mask_losses
        batch_depth_losses = torch.stack(self.depth_loss(pred_depth, true_depth))
        batch_silhouette_losses = torch.stack(self.silhouette_loss(pred_mask, true_mask))

        # # sample points within true depth images
        # true_points = self.renderer.sample_points_from_depth_images(true_depth, sample_ratio=0.05)
        # # get batch forward and backward chamfer distance
        # batch_chamfer_forward, batch_chamfer_backward = self.chamfer_loss(pred_vertices, true_points)

        # update loss_dict with mean batch_losses
        if 'depth_loss' in loss_dict: loss_dict['depth_loss'] += torch.mean(batch_depth_losses)
        else: loss_dict['depth_loss'] = torch.mean(batch_depth_losses)
        if 'silhouette_loss' in loss_dict: loss_dict['silhouette_loss'] += torch.mean(batch_silhouette_losses)
        else: loss_dict['silhouette_loss'] = torch.mean(batch_silhouette_losses)
        # return batch_losses
        return {'batch_depth_losses': batch_depth_losses, 'batch_silhouette_losses': batch_silhouette_losses}

    # depth loss between pred and true depth images
    def depth_loss(self, pred_depth, true_depth):
        # mask cloth regions
        depth_pred = pred_depth.clone()
        depth_pred[depth_pred == 255.] = 0.
        depth_true = true_depth.clone()
        depth_true[depth_true == 255.] = 0.
        # get depth difference
        depth_diff = torch.abs(depth_true - depth_pred)
        # calculate batch depth losses
        batch_depth_losses = [torch.sum(depth_diff[nb] / torch.sum(depth_true[nb])) for nb in range(depth_true.shape[0])]
        return batch_depth_losses

    # chamfer loss between pred vertices and true depth points
    def chamfer_loss(self, pred_vertices, true_points):
        # calculate batch forward and backward chamfer distance
        batch_chamfer_forward = [chamfer_distance(pred_vertices[nb], true_points[nb]) for nb in range(true_points.shape[0])]
        batch_chamfer_backward = [chamfer_distance(true_points[nb], pred_vertices[nb]) for nb in range(true_points.shape[0])]
        return batch_chamfer_forward, batch_chamfer_backward

    # silhouette loss between pred and true mask images
    def silhouette_loss(self, pred_mask, true_mask):
        # get mask difference
        mask_diff = torch.abs(true_mask - pred_mask)
        # calculate batch mask losses
        batch_mask_losses = [torch.sum(mask_diff[nb]) / torch.sum(true_mask[nb]) for nb in range(true_mask.shape[0])]
        return batch_mask_losses

    # weight pose alignment losses
    def weighted_train_losses(self, losses):
        # get pose align loss weight dict
        weight = {'vertex_loss': lambda cst: 10. ** 0 * cst,
                  'corner_loss': lambda cst: 10. ** -1 * cst,
                  'keypoint_loss': lambda cst: 10. ** -1 * cst,
                  'depth_loss': lambda cst: 10. ** 0 * cst,
                  'silhouette_loss': lambda cst: 10. ** 0 * cst,
                  'chamfer_loss': lambda cst: 10. ** 0 * cst,
                  }

        # init weight_loss with zero
        weight_losses = torch.tensor([0.]).to(self.device)
        # weight all loss
        for l in losses:
            weight_losses += weight[l](losses[l])
        return weight_losses

    # forward model and data for one epoch
    def forward(self, phase, data_loader, epoch=None, store_pred=False):
        # load model and opt
        model = self.model
        opt = self.opt

        # init is_training
        is_training = True if phase == 'train' else False
        # init pred_storage wirh names, meshes and indices
        pred_storage = {'real': {'names': [], 'meshes': [], 'indices': []}, 'simu': {'names': [], 'meshes': [], 'indices': []}}

        # loop over all batches
        for it, batch in enumerate(data_loader):
            # show process
            show_process(it, len(data_loader), prefix='{}/{}/{}/{}/Loss:{:.3f}'.format(phase, epoch, it*self.batch_size, len(data_loader)*self.batch_size, self.log_train_losses.avg))

            # zero the gradient of any prior passes
            opt.zero_grad()
            # send batch to device
            self.send_to_device(batch)
            # init loss_dict
            simu_loss_dict = {}
            real_loss_dict = {}
            # init batch_size
            batch_size = self.batch_size

            # calculate losses for image_simu and mesh_simu
            if 'image_simu' in batch:
                # get batch_size and augment_size
                batch_size, augment_size = batch['image_simu'].shape[:2]
                # forward model for augmented batch samples
                augment_pred_results = {'meshes': [], 'vertex_losses': [], 'pixel_losses': []}
                for na in range(augment_size):
                    # pred vertices form image_simu
                    simu_pred_meshes = model(batch['image_simu'][:, na])
                    augment_pred_results['meshes'].append(simu_pred_meshes)
                    # # update pixel_wise_losses with pred_mesh and image_simu: comment for faster train/val
                    if phase == 'test': augment_pred_results['pixel_losses'].append(self.pixel_wise_losses(simu_pred_meshes, batch['image_simu_input'][:, na], simu_loss_dict))
                    # update vertex_wise_losses with pred_mesh and mesh_simu
                    if 'mesh_simu' in batch:
                        augment_pred_results['vertex_losses'].append(self.vertex_wise_losses(simu_pred_meshes, batch['mesh_simu'][:, na], simu_loss_dict))
                # store best augmented pred_meshes
                if store_pred:
                    # get batch_pred_meshes (augment_size, batch_size, N, 3)
                    batch_pred_meshes = torch.stack([augment_pred_results['meshes'][na] for na in range(augment_size)])
                    # get batch_depth_losses and batch_silhouette_losses (augment_size, batch_size)
                    batch_depth_losses = torch.stack([augment_pred_results['pixel_losses'][na]['batch_depth_losses'] for na in range(augment_size)])
                    batch_silhouette_losses = torch.stack([augment_pred_results['pixel_losses'][na]['batch_silhouette_losses'] for na in range(augment_size)])
                    # find batch best indices and meshes according to pixel_losses
                    batch_best_pred_indices = torch.min(batch_depth_losses + batch_silhouette_losses, dim=0).indices
                    batch_best_pred_meshes = torch.stack([batch_pred_meshes[batch_best_pred_indices[nb], nb] for nb in range(batch_best_pred_indices.shape[0])])
                    # print('batch_best_pred_indices, batch_best_pred_meshes', batch_best_pred_indices.shape, batch_best_pred_meshes.shape)
                    pred_storage['simu']['names'] += batch['image_name']
                    pred_storage['simu']['meshes'].append(batch_best_pred_meshes)
                    pred_storage['simu']['indices'].append(batch_best_pred_indices)

            # calculate losses for image_real and mesh_real
            if 'image_real' in batch:
                # get batch_size and augment_size
                batch_size, augment_size = batch['image_real'].shape[:2]
                # forward model for augmented batch samples
                augment_pred_results = {'meshes': [], 'vertex_losses': [], 'pixel_losses': []}
                for na in range(augment_size):
                    # pred vertices form image_simu
                    real_pred_meshes = model(batch['image_real'][:, na])
                    augment_pred_results['meshes'].append(real_pred_meshes)
                    # update pixel_wise_losses with pred_mesh and image_real
                    augment_pred_results['pixel_losses'].append(self.pixel_wise_losses(real_pred_meshes, batch['image_real_input'][:, na], real_loss_dict))
                    # update vertex_wise_losses with pred_mesh and mesh_real
                    if 'mesh_real' in batch:
                        augment_pred_results['vertex_losses'].append(self.vertex_wise_losses(real_pred_meshes, batch['mesh_real'][:, na], real_loss_dict))
                # store best augmented pred_meshes
                if store_pred:
                    # get batch_pred_meshes (augment_size, batch_size, N, 3)
                    batch_pred_meshes = torch.stack([augment_pred_results['meshes'][na] for na in range(augment_size)])
                    # get batch_depth_losses and batch_silhouette_losses (augment_size, batch_size)
                    batch_depth_losses = torch.stack([augment_pred_results['pixel_losses'][na]['batch_depth_losses'] for na in range(augment_size)])
                    batch_silhouette_losses = torch.stack([augment_pred_results['pixel_losses'][na]['batch_silhouette_losses'] for na in range(augment_size)])
                    # find batch best indices and meshes
                    batch_best_pred_indices = torch.min(batch_depth_losses + batch_silhouette_losses, dim=0).indices
                    batch_best_pred_meshes = torch.stack([batch_pred_meshes[batch_best_pred_indices[nb], nb] for nb in range(batch_best_pred_indices.shape[0])])
                    # print('batch_best_pred_indices, batch_best_pred_meshes', batch_best_pred_indices.shape, batch_best_pred_meshes.shape)
                    pred_storage['real']['names'] += batch['image_name']
                    pred_storage['real']['meshes'].append(batch_best_pred_meshes)
                    pred_storage['real']['indices'].append(batch_best_pred_indices)

            # get weighted train losses
            simu_train_losses = self.weighted_train_losses(simu_loss_dict)
            real_train_losses = self.weighted_train_losses(real_loss_dict)
            train_losses = simu_train_losses + real_train_losses
            self.log_train_losses.update(train_losses.item(), batch_size)
            # optimize forward
            if is_training:
                # backward loss and optimize step
                train_losses.backward()
                opt.step()
                # adjust learning rate
                self.adjust_learning_rate(opt, epoch)
        # return averaged train losses, pred_storage
        return self.log_train_losses.avg, pred_storage

    # send batch data to device
    def send_to_device(self, batch):
        for k, v in batch.items():
            if k == 'image_name': continue
            batch[k] = v.to(self.device)

    # adjust learning rate at scheduler_step
    def adjust_learning_rate(self, opt, epoch):
        new_lr = self.lr * (0.1 ** (epoch // self.schedule_step))
        for param_group in opt.param_groups:
            param_group['lr'] = new_lr

    # train and val model with train and val dataloader
    def train_model(self, save_dir, train_loader, val_loader):
        # init train epoch, loss, model
        best_train_loss = float('inf')
        best_train_epoch = -1
        best_train_model = {}

        # forward mode for epoch_size
        for ne in range(self.epoch_size):
            # train model with train data_loader
            self.model.train()
            train_loss, _ = self.forward('train', train_loader, epoch=ne, store_pred=False)
            # validate model with val data_loader
            with torch.no_grad():
                self.model.eval()
                val_loss, _ = self.forward('val', val_loader, epoch=ne, store_pred=False)
                # update best train epoch
                if val_loss < best_train_loss:
                    best_train_epoch = ne
                    best_train_loss = val_loss
                    best_train_model = copy.deepcopy(self.model.state_dict())
            # save model with save step
            if ne % self.save_step == 0:
                torch.save({'epoch': ne, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.opt.state_dict()},
                           os.path.join(save_dir, f"model_{ne:04d}_{val_loss:.5f}.pt"))
        # save best model
        torch.save({'model_state_dict': best_train_model},
                   os.path.join(save_dir, f"bestmodel_{best_train_epoch:04d}_{best_train_loss:.5f}.pt"))

    # test model with test dataloader
    def test_model(self, checkpoint_fn, test_loader, augment_size):
        # load checkpoint to model
        checkpoint = torch.load(checkpoint_fn)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # test model with test data loader
        with torch.no_grad():
            self.model.eval()
            test_loss, pred_storage = self.forward('test', test_loader, store_pred=True)

        # process pred_storage = {'real': {'names': [], 'meshes': [], 'indices': []}, 'simu': {'names': [], 'meshes': [], 'indices': []}}
        for key, value in pred_storage.items():
            if len(pred_storage[key]['meshes']) > 0:
                # cat pred_meshes (IDS, N, 3) and pred_indices (IDS)
                pred_meshes = torch.cat(pred_storage[key]['meshes'], dim=0).cpu().numpy()
                pred_indices = torch.cat(pred_storage[key]['indices'], dim=0).cpu().numpy()

                # save pred_meshes for each test data
                os.makedirs(os.path.join(self.data_dir, key, 'pred'), exist_ok=True)
                for idx in range(pred_meshes.shape[0]):
                    # show save process
                    show_process(idx, pred_meshes.shape[0], prefix='save/{}/{}'.format(idx, pred_meshes.shape[0]))
                    # locate image_name
                    image_name = pred_storage[key]['names'][idx]
                    # read true depth image
                    true_depth_path = os.path.join(self.data_dir, key, 'test', '{}.{}_depth.png'.format(image_name, key))
                    true_depth = cv.imread(true_depth_path) if os.path.exists(true_depth_path) else None
                    # read true color image
                    true_color_path = os.path.join(self.data_dir, key, 'test', '{}.{}_color.png'.format(image_name, key))
                    true_color = cv.imread(true_color_path) if os.path.exists(true_color_path) else None
                    # rotate augmented prediction mesh back
                    pred_mesh = rotate_mesh(pred_meshes[idx], angle=-pred_indices[idx] * (360 // augment_size))
                    # assign pred mesh index according tp template mesh
                    pred_mesh = assign_mesh_index(pred_mesh, self.template_info["mesh_pos"])
                    # get mesh visible flag
                    pred_visible = cloth_visible_vertices(pred_mesh)
                    # append pred_mesh = [N, (position: 3, visible: 1)]
                    pred_mesh = np.append(pred_mesh, pred_visible.reshape(pred_mesh.shape[0], 1), axis=1)

                    # manipulate cloth prediction: manipulate_results[pred_template_image, pred_mesh_image, pred_group_image, pred_policy_image]
                    manipulate_results = manipulate_cloth_prediction(pred_mesh, true_depth, self.template_info, show=False)
                    save_pred_dir = os.path.join(self.data_dir, key, 'pred')
                    os.makedirs(save_pred_dir, exist_ok=True)
                    np.savetxt(os.path.join(save_pred_dir, '{}.pred_mesh.txt'.format(image_name)), pred_mesh)
                    cv.imwrite(os.path.join(save_pred_dir, '{}.pred_mesh.png'.format(image_name)), manipulate_results[1])
                    cv.imwrite(os.path.join(save_pred_dir, '{}.pred_group.png'.format(image_name)), manipulate_results[2])
                    cv.imwrite(os.path.join(save_pred_dir, '{}.pred_policy.png'.format(image_name)), manipulate_results[3])
                    if true_depth is not None: cv.imwrite(os.path.join(save_pred_dir, '{}.{}_depth.png'.format(image_name, key)), true_depth)
                    if true_color is not None: cv.imwrite(os.path.join(save_pred_dir, '{}.{}_color.png'.format(image_name, key)), true_color)

