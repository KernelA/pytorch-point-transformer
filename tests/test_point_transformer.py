from point_transformer import PointTransformerLayer, PointTransformerBlock

from .conftest import NUM_COORDS


def test_point_tranformer_layer(sample_batch):
    in_features = NUM_COORDS
    out_features = 16
    num_neighs = 6
    point_transformer = PointTransformerLayer(in_features=in_features,
                                              out_features=out_features, num_neighbors=num_neighs)

    new_features, _, _ = point_transformer((sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert new_features.shape == sample_batch.x.shape[:2] + (out_features, )


def test_transfromer_block(sample_batch):
    in_features = NUM_COORDS

    transformer_block = PointTransformerBlock(
        in_out_features=in_features, compress_dim=in_features * 2, num_neighbors=4)

    new_features, _, _ = transformer_block((sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert new_features.shape[:2] == sample_batch.x.shape[:2]
    assert new_features.shape[-1] == in_features
