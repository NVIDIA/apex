import unittest

import torch

import apex
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase


def init_model_and_optimizer():
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1.0)
    return model, optimizer


@unittest.skipUnless(torch.cuda.is_available(), "")
class TestDeprecatedWarning(unittest.TestCase):

    def test_amp(self):
        model, optimizer = init_model_and_optimizer()
        with self.assertWarns(apex.DeprecatedFeatureWarning):
            _ = apex.amp.initialize(model, optimizer)

    def test_fp16_model(self):
        model, _ = init_model_and_optimizer()
        with self.assertWarns(apex.DeprecatedFeatureWarning):
            _ = apex.fp16_utils.FP16Model(model)

    def test_fp16_optimizer(self):
        _, optimizer = init_model_and_optimizer()
        with self.assertWarns(apex.DeprecatedFeatureWarning):
            _ = apex.fp16_utils.FP16_Optimizer(optimizer)

    def test_fp16_loss_scaler(self):
        with self.assertWarns(apex.DeprecatedFeatureWarning):
             apex.fp16_utils.LossScaler()


class TestParallel(NcclDistributedTestBase):

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 2)

    def test_distributed_data_parallel(self):
        model, _ = init_model_and_optimizer()
        with self.assertWarns(apex.DeprecatedFeatureWarning):
            _ = apex.parallel.DistributedDataParallel(model)

    def test_convert_syncbn_model(self):
        model, _ = init_model_and_optimizer()
        with self.assertWarns(apex.DeprecatedFeatureWarning):
            _ = apex.parallel.convert_syncbn_model(model)


if __name__ == "__main__":
    unittest.main()
