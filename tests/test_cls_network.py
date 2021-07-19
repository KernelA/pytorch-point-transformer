from model import ClsPointTransformer

from .conftest import NUM_COORDS


def test_cls_networks(sample_batch):
    num_classes = 2
    simple_network = ClsPointTransformer(NUM_COORDS, num_classes)
    predicted_logits = simple_network(sample_batch.x, sample_batch.pos, sample_batch.batch)
    assert predicted_logits.shape == (sample_batch.num_graphs, num_classes)
