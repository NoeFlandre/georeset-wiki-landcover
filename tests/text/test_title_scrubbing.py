import pytest

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


@pytest.mark.parametrize(
    ("text", "title", "expected"),
    [
        ("  A \t B\n C  ", " \t ", "A B C"),
        ("  ...   et ...  ", " ... ", "ce lieu et ce lieu"),
        ("  A   B  ", "___", "A B"),
        ("  \n\t", "", ""),
        ("FORÊT_de_Test et Forêt / de / Test", "Forêt-de-Test", "ce lieu et ce lieu"),
        ("A+B et a / b", "A+B", "ce lieu et ce lieu"),
    ],
)
def test_remove_title_variants_edge_cases(text, title, expected):
    assert remove_title_variants(text, title) == expected
