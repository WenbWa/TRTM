import torch
import torch_scatter
import functools
import collections
import torch.nn as nn

# define edge and graph
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
GraphSet = collections.namedtuple('Graph', ['node_features', 'edge_set'])


# ------------------- Graph Attention Neural Network ------------------- #

class GATModel(nn.Module):
    """
    Basic Graph Attention Neural Network
    """
    def __init__(self, output_size, latent_size, num_layers, message_passing_steps):
        super(GATModel, self).__init__()
        # hyper-parameters
        self._output_size = output_size  # 3
        self._latent_size = latent_size  # 128
        self._num_layers = num_layers  # 2
        self._message_passing_steps = message_passing_steps  # 15
        # define encoder, updater, decoder
        self.encoder = Encoder(make_mlp=self._make_mlp)
        self.updater = Updater(make_mlp=self._make_mlp, output_size=self._latent_size, message_passing_steps=self._message_passing_steps)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False), output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """
            Build one MLP with output_size
        """
        # assign output_sizes
        if type(output_size) == int:
            output_sizes = [self._latent_size] * self._num_layers + [output_size]
        elif type(output_size) == list:
            output_sizes = output_size
        else:
            raise ValueError('Invalid output_size type')
        # construct MLP according to output_sizes
        network = LazyMLPBlock(output_sizes)
        # add norm layer
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=output_sizes[-1]))
        return network

    # forward GAT structure
    def forward(self, graph, image_feature):
        """
            Encode, Update, Decode Cloth Graph.
        """
        # encode template_graph and image_feature into latent_graph
        latent_graph = self.encoder(graph, image_feature)
        # update latent_graph with attention message passing
        latent_graph = self.updater(latent_graph)
        # decode latent_graph to mesh_position
        return self.decoder(latent_graph)


class Encoder(nn.Module):
    """
        Encode template node, edge features and image features into latent_graph
    """
    def __init__(self, make_mlp):
        super().__init__()
        self.node_encoder = make_mlp([256, 128])
        self.edge_encoder = make_mlp([256, 128])

    def forward(self, graph, image_feature):
        # get batch_size
        batch_size = image_feature.shape[0]
        # get number and dimension of graph node features
        node_num, node_dim = graph.node_features.shape
        # encode graph node_features and image_features (batch_size, node_num, 3 + 512) to latent node_features (batch_size, node_num, 128)
        node_feature = graph.node_features.view(1, node_num, node_dim).expand(batch_size, -1, -1)
        node_image_feature = image_feature.view(batch_size, 1, image_feature.shape[1]).expand(-1, node_num, -1)
        node_latents = self.node_encoder(torch.cat([node_feature, node_image_feature], -1))

        # encode graph edge features (batch_size, edge_num, 4) to latent edge features (batch_size, edge_num, 128)
        edge_num, edge_dim = graph.edge_set.features.shape
        edge_latents = self.edge_encoder(graph.edge_set.features.view(1, edge_num, edge_dim).expand(batch_size, -1, -1))
        # return encoded feature graph
        return GraphSet(node_latents, graph.edge_set._replace(features=edge_latents))


class Updater(nn.Module):
    """
        Update feature graph with N times of Attention Message Passing
    """
    def __init__(self, make_mlp, output_size, message_passing_steps):
        super().__init__()
        # stack GraphAttentionBlock with the number of message_passing_steps
        self._submodules_ordered_dict = collections.OrderedDict()
        for index in range(message_passing_steps):
            self._submodules_ordered_dict[str(index)] = GraphAttentionBlock(make_mlp=make_mlp, output_size=output_size)
        self.submodules = nn.Sequential(self._submodules_ordered_dict)

    def forward(self, graph):
        return self.submodules(graph)


class Decoder(nn.Module):
    """
        Decoder latent_graph to mesh_position
    """
    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.node_decoder = make_mlp(output_size)

    def forward(self, graph):
        return self.node_decoder(graph.node_features)


# ------------------- LazyMLP, GraphNet, Attention ------------------- #

class LazyMLPBlock(nn.Module):
    """
        Basic MLP structure
    """
    def __init__(self, output_sizes):
        super(LazyMLPBlock, self).__init__()
        # get MLP layers
        num_layers = len(output_sizes)
        self._layers_ordered_dict = collections.OrderedDict()
        # construct linear-relu MLP layers
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, x):
        y = self.layers(x)
        return y


class AttentionBlock(nn.Module):
    """
        Basic attention weight function.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, idx):
        input = self.linear(input)
        input = self.activation(input)
        # update attention_weights with softmax(MLP(edge_message_features))
        attention_weight = torch_scatter.composite.scatter_softmax(input, idx, dim=1)
        return attention_weight


class GraphAttentionBlock(nn.Module):
    """
        Basic Graph Attention Block with residual connections
    """
    def __init__(self, make_mlp, output_size):
        super(GraphAttentionBlock, self).__init__()
        # construct MLP models for edge and node
        self.edge_model = make_mlp(output_size)
        self.node_model = make_mlp(output_size)
        # construct attention weight model
        self.attention_model = AttentionBlock()

    def _update_edge_features(self, node_features, edge_set):
        """
            Aggregate node features, apply MLP edge function.
        """
        # get node_sender_features, node_receiver_features, edge_set.features
        node_sender_features = torch.index_select(input=node_features, dim=1, index=edge_set.senders)
        node_receiver_features = torch.index_select(input=node_features, dim=1, index=edge_set.receivers)
        features = [node_sender_features, node_receiver_features, edge_set.features]
        # update edge_message_features = edge_MLP([sender_node_features, receiver_node_features, edge_features])
        return self.edge_model(torch.cat(features, -1))

    def _update_node_features(self, node_features, edge_set):
        """
            Aggregate edge features, apply node function.
        """
        features = [node_features]
        # get learnable attention_weights
        attention_weights = self.attention_model(edge_set.features, edge_set.receivers)
        # get attention_message_features: attention_weights * edge_message_features
        features.append(torch_scatter.scatter_add(torch.mul(edge_set.features, attention_weights), edge_set.receivers, dim=1))
        # update node_message_features: node_MLP([receiver_node_features, attention_message_features])
        return self.node_model(torch.cat(features, -1))

    def forward(self, graph, residual=True):
        """
            Update Latent Graph with Attention Message Passing.
        """

        # apply edge functions: update edge_features
        updated_edge_features = self._update_edge_features(graph.node_features, graph.edge_set)
        new_edge_set = graph.edge_set._replace(features=updated_edge_features)

        # apply node functions: update node_features
        new_node_features = self._update_node_features(graph.node_features, new_edge_set)

        # apply residual change
        if residual:
            new_node_features += graph.node_features
            new_edge_set = new_edge_set._replace(features=new_edge_set.features + graph.edge_set.features)
        # return updated latent graph
        return GraphSet(new_node_features, new_edge_set)


