import numpy as np

from georeset.vision.image_encoder_registry import ENCODER_REGISTRY, _encoder_output_to_numpy


def test_image_encoder_registry_exposes_expected_model_configs_without_downloads() -> None:
    assert ENCODER_REGISTRY["clip_base"].model_name == "openai/clip-vit-base-patch32"
    assert ENCODER_REGISTRY["clip_large"].model_name == "openai/clip-vit-large-patch14"
    assert ENCODER_REGISTRY["dinov2_base"].model_name == "facebook/dinov2-base"


class _FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = np.asarray(values, dtype=np.float32)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeOutput:
    def __init__(self, pooler_output: _FakeTensor) -> None:
        self.pooler_output = pooler_output


def test_encoder_output_to_numpy_accepts_tensor_outputs() -> None:
    array = _encoder_output_to_numpy(_FakeTensor([[3.0, 4.0]]))

    np.testing.assert_allclose(array, np.asarray([[3.0, 4.0]], dtype=np.float32))


def test_encoder_output_to_numpy_accepts_huggingface_model_outputs() -> None:
    array = _encoder_output_to_numpy(_FakeOutput(_FakeTensor([[5.0, 12.0]])))

    np.testing.assert_allclose(array, np.asarray([[5.0, 12.0]], dtype=np.float32))
