import pytest
import torch

from point_transformer.blocks.segmentation import (
    DownBlock, MLPBlock, PairedBlock, PairedBlockWithSkipConnection, UpBlock)


def test_mlp_block(sample_batch):
    out_features = 3

    mlp_block = MLPBlock(in_features=sample_batch.x.shape[1],
                         out_features=out_features,
                         compress_dim=3,
                         num_neighbors=3,
                         is_jit=False)

    out_x, out_pos, out_bat = mlp_block((sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert out_x.shape[0] == out_pos.shape[0] == out_bat.shape[0]
    assert out_x.shape[1] == out_features
    assert (out_bat == sample_batch.batch).all()


def test_down_block(sample_batch):
    out_features = 4
    fps_sample_ratio = 0.5

    down_block = DownBlock(
        in_features=sample_batch.x.shape[1],
        out_features=out_features,
        compress_dim=3,
        num_neighbors=2,
        is_jit=False,
        fps_sample_ratio=fps_sample_ratio
    )

    out_x, out_pos, out_bat = down_block((sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert out_x.shape[1] == out_features
    assert out_x.shape[0] == sample_batch.x.shape[0] // 2


def test_up_block(sample_batch, sample_downsampled_batch):
    out_features = 3

    up_block = UpBlock(
        in_features=sample_batch.x.shape[1],
        out_features=out_features,
        compress_dim=2,
        num_neighbors=3,
        is_jit=False,
    )

    out_x, out_pos, out_bat = up_block(
        (sample_downsampled_batch.x, sample_downsampled_batch.pos, sample_downsampled_batch.batch,
         sample_batch.x, sample_batch.pos, sample_batch.batch)
    )

    torch.testing.assert_close(out_pos, sample_batch.pos)

    assert out_x.shape[0] == sample_batch.x.shape[0]
    assert out_x.shape[1] == out_features
    assert (out_bat == sample_batch.batch).all()


def test_paired_block_with_mlp(sample_batch):
    num_out_features = 3

    mlp_block = MLPBlock(in_features=sample_batch.x.shape[1],
                         out_features=num_out_features,
                         compress_dim=3,
                         num_neighbors=3,
                         is_jit=False)

    up_block = UpBlock(
        in_features=sample_batch.x.shape[1],
        out_features=num_out_features,
        compress_dim=2,
        num_neighbors=3,
        is_jit=False,
    )

    paired_block = PairedBlockWithSkipConnection(mlp_block, up_block)

    _ = paired_block.forward((sample_batch.x, sample_batch.pos,  sample_batch.batch))

    out_features, out_pos, out_batch = paired_block.forward(
        (sample_batch.x, sample_batch.pos, sample_batch.batch))

    assert out_features.shape[1] == num_out_features
    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0] == sample_batch.x.shape[0]


def test_paired_block_with_up_down(sample_batch):
    num_out_features = 3

    mlp_block = DownBlock(in_features=sample_batch.x.shape[1],
                          out_features=num_out_features,
                          compress_dim=3,
                          num_neighbors=3,
                          fps_sample_ratio=0.5,
                          is_jit=False)

    up_block = UpBlock(
        in_features=sample_batch.x.shape[1],
        out_features=num_out_features,
        compress_dim=2,
        num_neighbors=3,
        is_jit=False,
    )

    paired_block = PairedBlockWithSkipConnection(mlp_block, up_block)

    our_features_first, out_pos_first, out_batch_first = paired_block.forward(
        (sample_batch.x, sample_batch.pos,  sample_batch.batch))

    out_features, out_pos, out_batch = paired_block.forward(
        (our_features_first, out_pos_first, out_batch_first))

    assert out_features.shape[1] == num_out_features
    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0] == our_features_first.shape[0]
