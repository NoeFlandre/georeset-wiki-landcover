from georeset_wiki_landcover.vision.clip_transformers import select_clip_features


class _FakeClipOutput:
    def __init__(
        self,
        *,
        image_embeds: object | None = None,
        text_embeds: object | None = None,
        pooler_output: object | None = None,
    ) -> None:
        if image_embeds is not None:
            self.image_embeds = image_embeds
        if text_embeds is not None:
            self.text_embeds = text_embeds
        if pooler_output is not None:
            self.pooler_output = pooler_output


def test_select_clip_features_prefers_named_projection_outputs() -> None:
    projected = object()
    pooled = object()

    assert (
        select_clip_features(
            _FakeClipOutput(image_embeds=projected, pooler_output=pooled),
            "image_embeds",
        )
        is projected
    )
    assert (
        select_clip_features(
            _FakeClipOutput(text_embeds=projected, pooler_output=pooled),
            "text_embeds",
        )
        is projected
    )


def test_select_clip_features_falls_back_to_pooler_or_raw_tensor() -> None:
    pooled = object()
    raw = object()

    assert select_clip_features(_FakeClipOutput(pooler_output=pooled), "image_embeds") is pooled
    assert select_clip_features(raw, "text_embeds") is raw
