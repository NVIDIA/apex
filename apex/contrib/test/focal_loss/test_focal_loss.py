import unittest

import torch
import torch.nn.functional as F

reference_available = True
try:
    from torchvision.ops.focal_loss import sigmoid_focal_loss
except ImportError:
    reference_available = False

SKIP_TEST = None
try:
    from apex.contrib.focal_loss import focal_loss
except ImportError as e:
    SKIP_TEST = e


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
@unittest.skipIf(not reference_available, "Reference implementation `torchvision.ops.focal_loss.sigmoid_focal_loss` is not available.")
class FocalLossTest(unittest.TestCase):

    N_SAMPLES = 12
    N_CLASSES = 8
    ALPHA = 0.24
    GAMMA = 2.0
    REDUCTION = "sum"

    def test_focal_loss(self) -> None:
        if not reference_available:
            self.skipTest("This test needs `torchvision` for `torchvision.ops.focal_loss.sigmoid_focal_loss`.")
        else:
            x = torch.randn(FocalLossTest.N_SAMPLES, FocalLossTest.N_CLASSES).cuda()
            with torch.no_grad():
                x_expected = x.clone()
                x_actual = x.clone()
            x_expected.requires_grad_()
            x_actual.requires_grad_()

            classes = torch.randint(0, FocalLossTest.N_CLASSES, (FocalLossTest.N_SAMPLES,)).cuda()
            with torch.no_grad():
                y = F.one_hot(classes, FocalLossTest.N_CLASSES).float()

            expected = sigmoid_focal_loss(
                x_expected,
                y,
                alpha=FocalLossTest.ALPHA,
                gamma=FocalLossTest.GAMMA,
                reduction=FocalLossTest.REDUCTION,
            )

            actual = sum([focal_loss.FocalLoss.apply(
                x_actual[i:i+1],
                classes[i:i+1].long(),
                torch.ones([], device="cuda"),
                FocalLossTest.N_CLASSES,
                FocalLossTest.ALPHA,
                FocalLossTest.GAMMA,
                0.0,
            ) for i in range(FocalLossTest.N_SAMPLES)])

            # forward parity
            torch.testing.assert_close(expected, actual)

            expected.backward()
            actual.backward()

            # grad parity
            torch.testing.assert_close(x_expected.grad, x_actual.grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
