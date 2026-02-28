# Language Emergence Lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/emergent/blob/main/emergent_demo.ipynb)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License](https://img.shields.io/github/license/bangyen/emergent)](LICENSE)

**Multi-agent emergent language learning: Modular framework for studying communication protocols in fully reproducible referential games**

<p align="center">
  <!-- Note: Placeholder for future training progress visualization -->
</p>

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/bangyen/emergent.git
cd emergent
pip install -e .
pytest   # optional: run tests
langlab train --steps 1000 --k 5 --v 6
```

Or open in Colab: [Colab Notebook](https://colab.research.google.com/github/bangyen/emergent/blob/main/emergent_demo.ipynb).

## Results

| Metric | Value |
|--------|-------|
| Referential Accuracy | ~40-50% (Baseline) |
| Compositional Generalization | Supported |

## Features

- **Multi-Agent Communication** — Speaker-listener neural agents with discrete message generation using Gumbel-Softmax for differentiable training.
- **Flexible Architectures** — Support for MLP and sequence-based (RNN) agents with residual connections and layer normalization.
- **Comprehensive Analysis** — Tools for evaluation on IID and compositional splits, and research report generation.
- **Reproducible Research** — Seeded experiments and automated test validation.

## Repo Structure

```plaintext
emergent/
├── emergent_demo.ipynb  # Colab notebook demo
├── src/langlab/         # Core implementation
│   ├── core/            # Agent architectures and channel logic
│   ├── training/        # Training loops and grounding protocols
│   ├── data/            # World generation and datasets
│   ├── analysis/        # Evaluation and report generation
│   ├── apps/            # CLI interface
│   └── utils/           # Shared utilities
├── tests/               # Unit and integration tests
├── docs/                # Documentation and figures
└── outputs/             # Experiment results and checkpoints
```

## Validation

- ✅ Overall test coverage of ~70% (`pytest`)
- ✅ Reproducible seeds for experiments
- ✅ Core CLI functionality verified

## References

- [Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input](https://openreview.net/forum?id=HJGv1Z-AW) - Lazaridou et al. (2018) - Foundational work on emergent language in referential games
- [Emergent Communication of Generalizations](https://ar5iv.labs.arxiv.org/html/2106.02668) - Mu & Goodman (2021) - Generalizable communication protocols in referential games

## License

This project is licensed under the [MIT License](LICENSE).

