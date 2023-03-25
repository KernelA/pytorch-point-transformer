from torch.nn import Module

from ...types import PointSetBatchInfo


class PairedBlock(Module):
    def __init__(self, block, paired_block) -> None:
        super().__init__()
        self._is_init_pass = True
        self.first_pass = block
        self.final_pass = paired_block

    def forward(self, fpb_data: PointSetBatchInfo) -> PointSetBatchInfo:
        if self._is_init_pass:
            self._is_init_pass = False
            return self.forward_init(fpb_data)

        self._is_init_pass = True
        return self.forward_final(fpb_data)

    def forward_init(self, fpb_data: PointSetBatchInfo) -> PointSetBatchInfo:
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        return self.first_pass(fpb_data)

    def forward_final(self, fpb_data: PointSetBatchInfo) -> PointSetBatchInfo:
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        return self.final_pass(fpb_data)


class PairedBlockWithSkipConnection(PairedBlock):
    def __init__(self, block, paired_block) -> None:
        super().__init__(block=block, paired_block=paired_block)
        self.skip_connection_data = None

    def forward_init(self, fpb_data: PointSetBatchInfo) -> PointSetBatchInfo:
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        out = super().forward_init(fpb_data)
        self.skip_connection_data = out

        return out

    def forward_final(self, fpb_data: PointSetBatchInfo) -> PointSetBatchInfo:
        """ fpb_data:
            features [N x in_features] - point's features
            position [N x num_coords] - position of points. By default num_coords is equal to 3.
            batch [N x 1] - batch indices
        """
        assert self.skip_connection_data is not None

        args = fpb_data + self.skip_connection_data

        out = super().forward_final(args)

        self.skip_connection_data = None

        return out
