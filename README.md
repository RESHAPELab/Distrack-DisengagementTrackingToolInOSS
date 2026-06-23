---
title: Distrack
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Distrack - Developer Disengagement Tracking Tool

[![DOI](https://zenodo.org/badge/183011533.svg)](https://zenodo.org/badge/latestdoi/183011533)

Distrack is a research tool for extracting, analyzing, and visualizing developer activity in open-source software (OSS) projects on GitHub. It identifies disengagement patterns, computes project health metrics, and trains predictive models to flag developers at risk of leaving a project.

---

## Which path are you on?

| | **Reviewing the tool** | **Reproducing the study** |
|---|---|---|
| **Goal** | See the tool running on real data | Rebuild the full study from scratch |
| **Setup** | None | GitHub tokens, ~13 GB dataset |
| **Go to** | **[Live Demo](https://huggingface.co/spaces/SamUtz1/distrack)** | **[`REPRODUCTION.md`](REPRODUCTION.md)** |

**Reviewers:** open the [live demo](https://huggingface.co/spaces/SamUtz1/distrack) - no install, no tokens. Pick any pre-analyzed repository and explore the activity timelines, disengagement risk scores, and project-health views.

**Developers:** [`REPRODUCTION.md`](REPRODUCTION.md) walks through the full collect → analyze → model → serve pipeline, including GitHub tokens, rate-limit handling, and rebuilding the dataset from raw GitHub data.

---

## What is Distrack?

Distrack answers one question about an OSS project: **which contributors are drifting away, and how soon might they leave?** Point it at a GitHub project (or load a pre-analyzed one) and it gives you:

- **Per-developer activity timelines** - when each contributor was active, slowing down, or gone, plus a forecast of who is likely to stop contributing.
- **Disengagement simulations** - model a developer's departure and see how it might affect the repository.

Behind the scenes it runs a four-stage pipeline. Each stage writes its output to disk so the next can pick it up, and the dashboard reads the final results:

```
  GitHub API
      │
      ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. COLLECT   │──▶│ 2. ANALYZE   │──▶│ 3. MODEL     │──▶│ 4. SERVE     │
│ commits, PRs │   │ truck factor │   │ break detect │   │ FastAPI +    │
│ issues, tree │   │ STN, health  │   │ + prediction │   │ Streamlit UI │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

Full details of each stage - module layout, output files, and how to extend it, are in [`REPRODUCTION.md`](REPRODUCTION.md).

---

## Research Context

The project began when Igor Steinmacher read *"Will you come back to contribute?"*, the paper that supplies the developer state labeler. An NSF Small proposal followed in 2023; after that, a NAU internal TRIF Faculty Research grant funded the work, which began in December 2024. The first results paper was submitted in January 2026 to the ACM Student Research Competition at ICSE (2nd place), and a tool-demonstration paper followed at ICSME 2026.

Publication and funding trail (chronological):

- *Will you come back to contribute? Investigating the inactivity of OSS developers* (2021) - [`Docs/1. 2021_2103.04656v3.pdf`](Docs/1.%202021_2103.04656v3.pdf)
- *Sustainability Breaks in OSS* - NSF Small Grant (2023) - [`Docs/2. 2023_NSF__Small__Sustainability_Breaks.pdf`](Docs/2.%202023_NSF__Small__Sustainability_Breaks.pdf)
- *NAU TRIF Grant - Sustainability Breaks* (2024) - [`Docs/3. 2024_NAU_Grant_Sustainability_Breaks.pdf`](Docs/3.%202024_NAU_Grant_Sustainability_Breaks.pdf)
- *Early Forecasting of Developer Inactivity in Open Source Projects* - ICSE 2026 SRC - [`Docs/4. 2025_SRC2026_Sam.pdf`](Docs/4.%202025_SRC2026_Sam.pdf)
- *DisTrac: Disengagement Tracking Tool in Open Source Projects* - ICSME 2026 - [`Docs/5. 2026_ICSME2026_Distrac_Sam.pdf`](Docs/5.%202026_ICSME2026_Distrac_Sam.pdf)

---

## Citation

If you use Distrack in your research, please cite:

```bibtex
@inproceedings{distrack2026,
  title     = {DisTrac: A Disengagement Tracking Tool for OSS},
  author    = {Utzinger, Samuel and others},
  booktitle = {Proceedings of ICSME 2026},
  year      = {2026}
}
```
