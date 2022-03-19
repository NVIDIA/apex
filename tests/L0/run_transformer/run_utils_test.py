import torch

from apex.transformer.tensor_parallel import utils


def test_divide():
    assert utils.divide(8, 4) == 2


def test_split_tensor_along_last_dim():
    inputy = torch.randn((100, 100, 100))
    splits = utils.split_tensor_along_last_dim(inputy, 10)
    last_dim_shapes = torch.tensor([int(split.size()[-1]) for split in splits])
    assert torch.equal(last_dim_shapes, torch.full((10,), 10))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    test_divide()
    test_split_tensor_along_last_dim()
    print(">> passed the test :-)")
