#GeoReset

This repository is part of the GeoReset project: https://geo-reset.sylvainlobry.com/

This codebase focuses on the text pipeline and is meant as a place of experimentations.

For now our goal is to implement the following pipeline:

- Choose a map (CORINE? EUNIS? LUCAS?)
- Choose a region (Alsace?)
- Scrap polygones and their classes from the map
- Scrap relevant web pages with respect to these polygons
- Ask an LLM to summarize / rephrase these web pages
- Make sure there is no mention of the place itself in the resulting descriptions
- Give the rephrased descriptions to another LLM and ask it to classify the associated polygon (classes will be provided to the LLM)
- We evaluate the classification produced by the LLM using standard metrics like precision, recall and so on.

The goal here is to assess whether based solely on the text description an LLM is able to deduce the correct class of the polygon.