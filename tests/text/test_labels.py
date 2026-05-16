from georeset.text.labels import EVIDENCE_TYPE_LABELS, enum_text, format_list


def test_enum_text_uses_label_map_and_falls_back_to_readable_token() -> None:
    assert enum_text("forest", EVIDENCE_TYPE_LABELS) == "forêt"
    assert enum_text("urban_or_artificial", {}) == "urban or artificial"
    assert enum_text("", EVIDENCE_TYPE_LABELS) == "inconnue"
    assert enum_text(None, EVIDENCE_TYPE_LABELS) == "inconnue"


def test_format_list_uses_none_word_for_empty_values() -> None:
    assert format_list(["forêt", "eau"]) == "forêt, eau"
    assert format_list([]) == "aucun"
