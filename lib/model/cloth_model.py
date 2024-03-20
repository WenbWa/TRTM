import torch
import pickle
import numpy as np
import torch.nn as nn
from lib.model import gat_model, resnet_model


class ClothModel(nn.Module):
    """
        Template-based Mass-spring Cloth GNN
    """
    def __init__(self, template_info, message_passing_steps=15):
        super(ClothModel, self).__init__()

        # init template_info
        self.template_info = template_info
        self.template_mesh_pos = self.template_info['mesh_pos']
        self.template_edge_idx = self.template_info['edge_idx'].astype(int)

        # init backbone ResNet model
        self.backbone_model = resnet_model.get_model('resnet34')

        # init learned GAT model
        self.learned_model = gat_model.GATModel(
            output_size=3,
            latent_size=128,
            num_layers=2,
            message_passing_steps=message_passing_steps
        )

    # build template_graph with template_nodes, edge_senders, and edge_receivers
    def _build_template_graph(self):
        # init node features as template mesh positions: N x 3
        node_features = torch.from_numpy(self.template_mesh_pos).float().cuda()

        # create two-way connectivity of edge senders and receivers
        senders = self.template_edge_idx[:, 0]
        receivers = self.template_edge_idx[:, 1]
        sender = torch.from_numpy(np.concatenate([senders, receivers], 0)).to(torch.int64).cuda()
        receiver = torch.from_numpy(np.concatenate([receivers, senders], 0)).to(torch.int64).cuda()
        # assign edge features as edge vertices' relative coordinate and norm: E x 4
        relative_edge_vector = (torch.index_select(node_features, 0, sender) - torch.index_select(node_features, 0, receiver))
        relative_edge_norm = torch.norm(relative_edge_vector, dim=-1, keepdim=True)
        edge_features = torch.cat((relative_edge_vector, relative_edge_norm), -1)

        # return gat model with node features and mesh edges
        mesh_edges = gat_model.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receiver,
            senders=sender
        )
        return gat_model.GraphSet(node_features=node_features, edge_set=mesh_edges)

    def forward(self, batch):
        # resnet encode image feature
        image_feature = self.backbone_model(batch)
        # build template graph from template_info
        template_graph = self._build_template_graph()
        # forward gat with message_passing_steps
        pred_mesh = self.learned_model(template_graph, image_feature)
        return pred_mesh

