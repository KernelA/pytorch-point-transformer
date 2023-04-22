import torch

from point_transformer import TransitionUp

from .conftest import NUM_POINTS


@torch.no_grad()
def test_transition_up(sample_batch):
    in_features = sample_batch.x.shape[1]
    out_features = 8

    in_features_original = in_features // 2

    transition_up = TransitionUp(
        in_features=in_features,
        out_features=out_features,
        in_features_original=in_features_original)

    sample_size = max(NUM_POINTS // 2, 1)
    downsample_features = sample_batch.x[:sample_size, ...]
    downsample_pos = sample_batch.pos[:sample_size, ...]
    downsample_batch = sample_batch.batch[:sample_size, ...]

    upsampled_features, upsampled_positions, _ = transition_up(
        (downsample_features, downsample_pos, downsample_batch,
         sample_batch.x[:, :in_features_original], sample_batch.pos, sample_batch.batch)
    )

    assert upsampled_features.shape[1] == out_features
    assert upsampled_features.shape[0] == sample_batch.x.shape[0]
    assert upsampled_positions.shape == sample_batch.pos.shape
