from georeset.utils.articles import index_articles_by_pageid


def test_index_articles_by_pageid_normalizes_pageids_and_preserves_first_duplicate():
    first = {"pageid": 100, "title": "First"}
    duplicate = {"pageid": "100", "title": "Duplicate"}
    second = {"pageid": 200, "title": "Second"}

    result = index_articles_by_pageid([first, duplicate, second])

    assert result == {"100": first, "200": second}


def test_index_articles_by_pageid_skips_missing_pageids():
    article_without_pageid = {"title": "No id"}
    article_with_none_pageid = {"pageid": None, "title": "None id"}

    result = index_articles_by_pageid([article_without_pageid, article_with_none_pageid])

    assert result == {}
