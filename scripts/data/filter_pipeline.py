"""Repository wrapper for the packaged data-filtering CLI."""

from georeset_wiki_landcover.cli.data.filter_pipeline import *  # noqa: F403
from georeset_wiki_landcover.cli.data.filter_pipeline import main

if __name__ == "__main__":
    main()
