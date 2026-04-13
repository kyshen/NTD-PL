import numpy as np

from src.metrics import val_SAM
from src.tasks.decompose import DecomposeTask
from src.types import Tensor


def test_sam_is_zero_for_identical_spectra():
    x = np.ones((4, 4, 3), dtype=np.float32)
    original = Tensor(shape=x.shape, dense=x)
    reconstructed = Tensor(shape=x.shape, dense=x.copy())

    assert np.isclose(val_SAM(original, reconstructed), 0.0)


def test_sam_is_ninety_degrees_for_orthogonal_spectra():
    x = np.zeros((1, 1, 2), dtype=np.float32)
    y = np.zeros((1, 1, 2), dtype=np.float32)
    x[0, 0] = [1.0, 0.0]
    y[0, 0] = [0.0, 1.0]

    original = Tensor(shape=x.shape, dense=x)
    reconstructed = Tensor(shape=y.shape, dense=y)

    assert np.isclose(val_SAM(original, reconstructed), 90.0)


def test_decompose_task_emits_sam():
    x = np.ones((8, 8, 4), dtype=np.float32)
    rec = x.copy()
    rec[0, 0] = [0.5, 1.0, 1.0, 1.0]

    class DummyData:
        def get(self, split: str):
            return Tensor(shape=x.shape, dense=x)

        def get_size(self):
            return x.size

    class DummyMethod:
        def reconstruct(self):
            return Tensor(shape=rec.shape, dense=rec)

        def get_num_params(self):
            return 10

    task = DecomposeTask(log_level=0)
    task.data = DummyData()
    task.method = DummyMethod()
    task.fit_time_sec = 0.5

    eval_dict = task.evaluate()

    assert "SAM" in eval_dict
    assert eval_dict["SAM"] > 0.0
