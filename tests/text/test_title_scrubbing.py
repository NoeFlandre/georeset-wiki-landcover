from georeset_wiki_landcover.text.title_scrubbing import remove_title_variants


def test_remove_title_variants_masks_exact_and_separator_title_forms() -> None:
    assert (
        remove_title_variants(
            "Forêt-de-Test contient des boisements. La Forêt de Test est humide.",
            "Forêt-de-Test",
        )
        == "ce lieu contient des boisements. La ce lieu est humide."
    )


def test_remove_title_variants_collapses_whitespace_and_ignores_empty_titles() -> None:
    assert remove_title_variants("A   B", "") == "A B"
