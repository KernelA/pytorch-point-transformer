import torch
from torch import nn
from torch_geometric.nn import fps, knn, max_pool_x

from ..types import PointSetBatchInfo


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
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )
        self._out_features = out_features

    def forward(self, position: torch.Tensor, features: torch.Tensor, batch_indices: torch.LongTensor) -> PointSetBatchInfo:
        """
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        fps_indices = fps(position, batch=batch_indices, ratio=self.fps_sample_ratio)
        out_features = self.mlp(features).view(-1, self._out_features)

        new_batch_indices = batch_indices[fps_indices]

        # [2 x num_neighbors * fps_indices]
        # self point in knn
        nearest_indices = knn(
            features, features[fps_indices],
            k=self.num_neighbors,
            batch_x=batch_indices,
            batch_y=new_batch_indices)

        new_features = max_pool_x(
            cluster=nearest_indices[0, :], x=out_features[nearest_indices[1, :]], batch=batch_indices[nearest_indices[1, :]])[0]

        return new_features, position[fps_indices], new_batch_indices
