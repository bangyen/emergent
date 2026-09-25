# Language Emergence Lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/emergent/blob/main/emergent_demo.ipynb)
[![CI](https://github.com/bangyen/emergent/actions/workflows/ci.yml/badge.svg)](https://github.com/bangyen/emergent/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/bangyen/emergent)](LICENSE)

**Multi-agent emergent language learning: Modular framework for studying communication protocols in fully reproducible referential games**

<p align="center">
  <img src="docs/training_curve.png" alt="Training curve: iid accuracy reaches 100%, compositional accuracy ~90%" width="640">
</p>

## Quickstart

```bash
git clone https://github.com/bangyen/emergent.git
cd emergent
pip install -e ".[analysis]"   # or: uv sync --extra dev
langlab train --steps 10000 --heldout red,circle   # writes outputs/metrics.csv + checkpoint
langlab eval --ckpt outputs/checkpoints/final_model.pt --split compo_target
langlab plot                                        # outputs/training_curve.png
```

Or open in Colab: [Colab Notebook](https://colab.research.google.com/github/bangyen/emergent/blob/main/emergent_demo.ipynb).

## Results

Every red circle is held out of training (in the large world, every yellow star). 10k steps, `k=5` objects per scene so chance is 20%, vocabulary 16, message length 2; mean ± std over seeds 1–3.

<!-- results:start -->
| Run | Seeds | IID acc | Compo acc | Held-out target acc | Worst pair acc | Agreement | TopSim | PosDis | # messages |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 3 | 99.5 ± 0.8% | 90.3 ± 3.2% | 56.4 ± 16.1% | – | – | 0.29 ± 0.02 | 0.19 ± 0.04 | 16.3 ± 1.2 |
| GRU | 3 | 99.5 ± 0.9% | 85.1 ± 3.4% | 34.5 ± 18.2% | – | – | 0.37 ± 0.08 | 0.12 ± 0.06 | 16.7 ± 0.6 |
| MLP, population of 3 with turnover | 3 | 94.0 ± 3.2% | 87.0 ± 2.3% | 57.1 ± 12.7% | 90.0 ± 6.5% | 3.1 ± 2.8% | 0.35 ± 0.04 | 0.20 ± 0.06 | 13.0 ± 2.1 |
| MLP, large world | 3 | 94.6 ± 1.6% | 93.3 ± 1.9% | 78.0 ± 5.5% | – | – | 0.39 ± 0.03 | 0.07 ± 0.08 | 52.3 ± 11.6 |
<!-- results:end -->

- **IID**: fresh scenes with no held-out objects. **Compo**: scenes that *contain* a held-out object. **Held-out target**: the target itself is a held-out object — the strict compositional test.
- **TopSim** / **PosDis**: topographic similarity and positional disentanglement of the speaker's lexicon (1.0 = perfectly compositional). **# messages**: distinct messages across all objects (18 in the default world, 225 in the large one).
- **Population only**: accuracy is averaged over every speaker × listener pair; **worst pair** is the weakest pair and **agreement** is how often two speakers use the same message for an object.

What this shows:

- Agents almost always solve the game on familiar objects, but the strict test tells a different story: when the target is an unseen combination, accuracy drops to 35–57% in the default world. The looser "compo" score (85–90%) hides this, because most of those scenes can be solved without naming the held-out object.
- The languages are only weakly compositional (TopSim ≈ 0.3–0.4).
- The large world generalizes better (78% on held-out targets), which fits the idea that more combinations push the agents toward compositional codes.
- In populations, speakers almost never agree on messages (about 3%), yet every listener understands every speaker (the worst pair still scores 90%). The listeners become multilingual rather than the community converging on one language.

Regenerate this table with `langlab sweep experiments/readme.json --jobs 4 --readme README.md`.

## Features

- **Multi-Agent Communication** — Speaker and Listener agents exchange discrete messages; the Speaker is trained with REINFORCE (Gumbel-max sampling, moving-average baseline, entropy bonus), the Listener with cross-entropy.
- **Flexible Architectures** — MLP agents with residual connections and layer normalization, or GRU sequence agents (`--use-sequence-models`).
- **Compositional Generalization** — `--heldout` removes attribute combinations from the training stream; training periodically evaluates on IID, compositional and held-out-target sets.
- **Language Metrics** — topographic similarity, positional disentanglement, message entropy and lexicon size, logged at every evaluation.
- **Populations** — `langlab pop-train` trains speakers and listeners in random pairings, with optional generational turnover (`--lifespan`).
- **Pragmatics** — `langlab eval --split distractor --pragmatic` evaluates an RSA pragmatic listener on scenes with similar distractors.
- **Worlds** — `--world large` switches from 18 objects (3 colours × 3 shapes × 2 sizes) to 225 (5 × 5 × 3 × 3 textures).
- **Sweeps** — `langlab sweep config.json` runs variants × grid × seeds, resumes interrupted runs, and aggregates results into a Markdown/CSV table.

## Roadmap

Done:

- [x] Language structure metrics (TopSim, PosDis, message entropy)
- [x] Strict compositional split where the target itself is held out
- [x] Pragmatic listener and distractor scenes wired into `eval`; unused channel code removed
- [x] Tuned GRU sequence agents (entropy weight 0.05 brings them level with the MLP on IID accuracy)
- [x] Config-driven sweeps that regenerate the results table
- [x] Configurable attribute spaces (`--world large`)
- [x] Population training with generational turnover

Next:

- [ ] Close the held-out-target gap: improve the 35–57% accuracy on unseen combinations, for example with more distractors that share attributes with the target, or a smaller vocabulary
- [ ] Iterated-learning experiments: track TopSim across generations and against the population size
- [ ] Population convergence: find out why speakers keep separate languages (agreement is about 3%) and test whether shared listeners or a bigger population change that
- [ ] Variable-length messages with an end-of-sequence token and a length cost
- [ ] Pixel inputs in place of one-hot attribute vectors

## Repo Structure

```plaintext
emergent/
├── emergent_demo.ipynb  # Colab notebook demo
├── src/langlab/         # Core implementation
│   ├── core/            # Agent architectures and channel logic
│   ├── training/        # Training loop, population training, sweeps
│   ├── data/            # World generation and datasets
│   ├── analysis/        # Evaluation, language metrics, reports, plotting (needs the `analysis` extra)
│   ├── apps/            # CLI interface
│   └── utils/           # Shared utilities
├── experiments/         # Sweep configs (experiments/readme.json builds the results table)
├── tests/               # Unit and integration tests
├── docs/                # Documentation and figures
└── outputs/             # Experiment results and checkpoints
```

## Validation

- CI runs ruff, mypy and the test suite on Python 3.10–3.12, then a CLI smoke run (train → eval → plot)
- Test coverage above 80% (`pytest --cov=src`)

## References

- [Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input](https://openreview.net/forum?id=HJGv1Z-AW) - Lazaridou et al. (2018) - Foundational work on emergent language in referential games
- [Compositionality and Generalization in Emergent Languages](https://arxiv.org/abs/2004.09124) - Chaabouni et al. (2020) - Positional disentanglement and generalization
- [Emergent Communication of Generalizations](https://ar5iv.labs.arxiv.org/html/2106.02668) - Mu & Goodman (2021) - Generalizable communication protocols in referential games

## License

This project is licensed under the [MIT License](LICENSE).

