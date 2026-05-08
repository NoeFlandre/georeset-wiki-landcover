# Documentation

```
docs/
├── guides/
│   └── gpu_classification_guide.md   # Grid5000 GPU job submission guide
├── diagrams/
│   ├── pipeline_diagram.tex          # TikZ source for pipeline overview
│   ├── pipeline_diagram.pdf          # Compiled PDF
│   └── pipeline_diagram-1.png        # PNG render for README
└── index.md                          # This file
```

## Diagrams

The pipeline overview diagram (`diagrams/pipeline_diagram-1.png`) is built from `diagrams/pipeline_diagram.tex`. To recompile:

```bash
cd docs/diagrams
pdflatex pipeline_diagram.tex
pdftoppm -png -r 150 pipeline_diagram.pdf pipeline_diagram
```

## Guides

See `guides/gpu_classification_guide.md` for Grid5000 job submission documentation.
