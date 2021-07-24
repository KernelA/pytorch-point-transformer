import pytest

from point_transformer import LinearBN

from .conftest import NUM_COORDS


def test_cls_networks(sample_batch):
    out_features = 5
    layer = LinearBN(NUM_COORDS, out_features)
    output = layer(sample_batch.x)

    assert output.shape == sample_batch.x.shape[:2] + (out_features,)
