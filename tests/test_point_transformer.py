import torch
from point_transformer import PointTransformerLayer, PointTransformerBlock

from .conftest import NUM_COORDS


@torch.no_grad()
def test_point_transformer_layer(sample_batch):
    in_features = NUM_COORDS
    out_features = 8
    num_neighs = 3
    point_transformer = PointTransformerLayer(in_features=in_features,
                                              out_features=out_features, num_neighbors=num_neighs)

    new_features, new_pos, new_batch = point_transformer(
        (sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert new_features.shape == (sample_batch.x.shape[0],) + (out_features, )
    torch.testing.assert_close(new_pos, sample_batch.pos)
    torch.testing.assert_close(new_batch, sample_batch.batch)


@torch.no_grad()
def test_jit_transformer_layer(sample_batch):
    in_features = NUM_COORDS
    out_features = 8
    num_neighs = 6
    point_transformer = PointTransformerLayer(in_features=in_features,
                                              out_features=out_features, num_neighbors=num_neighs).jittable()

    point_transformer = torch.jit.script(point_transformer)

    point_transformer((sample_batch.x, sample_batch.pos, sample_batch.batch))


@torch.no_grad()
def test_transformer_block(sample_batch):
    in_features = NUM_COORDS

    transformer_block = PointTransformerBlock(
        in_out_features=in_features, compress_dim=in_features * 2, num_neighbors=4)

    new_features, new_pos, new_batch = transformer_block(
        (sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert new_features.shape[0] == sample_batch.x.shape[0]
    torch.testing.assert_close(new_pos, sample_batch.pos)
    torch.testing.assert_close(new_batch, sample_batch.batch)
    assert new_features.shape[-1] == in_features
