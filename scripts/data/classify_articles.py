"""Repository wrapper for the packaged article classification CLI."""

from georeset_wiki_landcover.cli.data.classify_articles import *  # noqa: F403
from georeset_wiki_landcover.cli.data.classify_articles import main

if __name__ == "__main__":
    main()
