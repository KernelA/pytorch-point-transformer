from torch import nn
from torch_geometric.nn import fps, knn, max_pool_x

from ..types import PointSetBatchInfo
from ..layers import LinearBN


class TransitionDown(nn.Module):
    def __init__(self,
                 *,
                 in_features: int,
                 out_features: int,
                 num_neighbors: int,
                 fps_sample_ratio: float):
        super().__init__()
        self.fps_sample_ratio = fps_sample_ratio
        self.num_neighbors = num_neighbors
        self.mlp = LinearBN(in_features, out_features)
        self._out_features = out_features

    def forward(self, input: PointSetBatchInfo) -> PointSetBatchInfo:
        """input contains:
            features [B x N x in_features] - node features
            positions [B x N x num_coords] - position of points. By default num_coords is equal to 3.
            batch - batch indices
        """
        features, positions, batch = input
        batch_size = positions.shape[0]
        num_coords = positions.shape[-1]
        flatten_positions = positions.view(-1, num_coords)
        fps_indices = fps(flatten_positions, batch=batch, ratio=self.fps_sample_ratio)

        num_features = features.shape[-1]
        flatten_features = features.view(-1, num_features)

        out_features = self.mlp(features).reshape(-1, self._out_features)

        # [2 x num_neighbors * fps_indices]
        # self point in knn
        nearthest_indices = knn(
            flatten_features, flatten_features[fps_indices], k=self.num_neighbors, batch_x=batch, batch_y=batch[fps_indices])

        new_features = max_pool_x(
            cluster=nearthest_indices[0, :], x=out_features[nearthest_indices[1, :]], batch=batch[nearthest_indices[1, :]])[0]

        return new_features.view(batch_size, -1, self._out_features), flatten_positions[fps_indices].view(batch_size, -1, num_coords), batch[fps_indices]
