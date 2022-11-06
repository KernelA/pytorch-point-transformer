import torch

from point_transformer.models import ClsPointTransformer

from .conftest import NUM_COORDS


# TODO: Result of jitted model is different than original
@torch.no_grad()
def test_jit(sample_batch):
    num_classes = 2
    simple_network = ClsPointTransformer(
        NUM_COORDS, num_classes, num_transformer_blocks=1, is_jit=True)

    jitted_network = torch.jit.script(simple_network)

    predicted_logits = jitted_network(
        (sample_batch.x, sample_batch.pos, sample_batch.batch)
    )

    assert predicted_logits.shape == (sample_batch.num_graphs, num_classes)
