from torch.nn import Module, Sequential

from ...blocks import PointTransformerBlock, TransitionDown, TransitionUp
from ...layers import MLP
from ...types import PointSetBatchInfo, TwoInputsType


class MLPBlock(Module):
    def __init__(self, *, in_features: int, out_features: int, compress_dim: int, num_neighbors: int,
                 is_jit: bool) -> None:
        super().__init__()

        self.forward_modules = Sequential(
            MLP(in_features, out_features),
            PointTransformerBlock(in_out_features=out_features, compress_dim=compress_dim,
                                  num_neighbors=num_neighbors, is_jit=is_jit),
        )

    def forward(self, fpb_data: PointSetBatchInfo):
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        return self.forward_modules(fpb_data)


class DownBlock(Module):
    def __init__(self, *,
                 in_features: int,
                 out_features: int,
                 compress_dim: int,
                 num_neighbors: int,
                 is_jit: bool,
                 fps_sample_ratio: float = 0.25) -> None:
        super().__init__()

        self.forward_modules = Sequential(
            TransitionDown(
                in_features=in_features,
                out_features=out_features,
                num_neighbors=num_neighbors,
                fps_sample_ratio=fps_sample_ratio
            ),
            PointTransformerBlock(in_out_features=out_features,
                                  compress_dim=compress_dim, num_neighbors=num_neighbors, is_jit=is_jit)
        )

    def forward(self, fpb_data: PointSetBatchInfo):
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        return self.forward_modules(fpb_data)


class UpBlock(Module):
    def __init__(self, *,
                 in_features: int,
                 in_features_original: int,
                 out_features: int,
                 compress_dim: int,
                 num_neighbors: int,
                 is_jit: bool,
                 ) -> None:
        super().__init__()

        self.up = TransitionUp(
            in_features=in_features,
            out_features=out_features,
            in_features_original=in_features_original
        )
        self.transformer_block = PointTransformerBlock(in_out_features=out_features,
                                                       compress_dim=compress_dim,
                                                       num_neighbors=num_neighbors,
                                                       is_jit=is_jit)

    def forward(self, fpb_data: TwoInputsType) -> PointSetBatchInfo:
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        out_features, out_pos, out_batch = self.up(fpb_data)
        return self.transformer_block((out_features, out_pos, out_batch))
