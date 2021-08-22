from point_transformer import TransitionDown

from .conftest import NUM_COORDS


def test_transition_down(sample_batch):
    in_features = NUM_COORDS
    out_features = 16
    transition_down = TransitionDown(
        in_features=in_features, out_features=out_features, num_neighbors=10, fps_sample_ratio=0.5)

    new_features, new_postions, new_batch = transition_down(
        (sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert new_features.shape[-1] == out_features
    assert new_features.shape[:2] == new_postions.shape[:2]
    assert new_features.shape[1] == sample_batch.x.shape[1] // 2
