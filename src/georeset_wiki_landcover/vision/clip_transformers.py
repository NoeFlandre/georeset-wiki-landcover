"""Small typed adapters around optional Transformers CLIP outputs."""

from __future__ import annotations

from typing import Literal, Protocol, cast


class TorchFeatureTensor(Protocol):
    def norm(self, dim: int, keepdim: bool) -> TorchFeatureTensor: ...

    def detach(self) -> TorchFeatureTensor: ...

    def cpu(self) -> TorchFeatureTensor: ...

    def numpy(self) -> object: ...

    def __truediv__(self, other: object) -> TorchFeatureTensor: ...


ClipFeatureAttribute = Literal["image_embeds", "text_embeds"]


def select_clip_features(
    output: TorchFeatureTensor | object,
    preferred_attribute: ClipFeatureAttribute,
) -> TorchFeatureTensor:
    projected = getattr(output, preferred_attribute, None)
    if projected is not None:
        return cast(TorchFeatureTensor, projected)
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return cast(TorchFeatureTensor, pooler_output)
    return cast(TorchFeatureTensor, output)
