from point_transformer import PointTransformerLayer, PointTransformerBlock

from .conftest import NUM_COORDS, NUM_POINTS


def test_point_tranformer_layer(sample_batch):
    in_features = NUM_COORDS
    out_features = 16
    num_neighs = 6
    point_transformer = PointTransformerLayer(in_features=in_features,
                                              out_features=out_features, num_neighbors=num_neighs)

    output = point_transformer(sample_batch.x, sample_batch.pos, sample_batch.batch)

    assert output.shape == (sample_batch.x.shape[0], out_features)


def test_transfromer_block(sample_batch):
    in_features = NUM_COORDS

    transformer_block = PointTransformerBlock(
        in_features=in_features, compress_dim=in_features * 2, num_neighbors=4)

    new_features = transformer_block(sample_batch.x, sample_batch.pos, sample_batch.batch)

    assert new_features.shape[0] == sample_batch.x.shape[0]
    assert new_features.shape[1] == in_features
