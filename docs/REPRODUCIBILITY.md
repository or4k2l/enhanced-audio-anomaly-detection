# Reproducibility

## Supported Environment

Continuous integration validates Python 3.10, 3.11, and 3.12 on Ubuntu. Use one
of these versions for development and benchmark runs. The runtime dependencies
are constrained in `requirements.txt`; development tools are kept separately in
`requirements-dev.txt`.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
pytest -q
```

To preserve the exact environment used for an experiment, record the resolved
packages next to its output:

```bash
python -m pip freeze > experiments/<run-name>/requirements.lock
```

Recreate that run with `python -m pip install -r
experiments/<run-name>/requirements.lock`. A lock file is specific to its Python
version, operating system, and hardware-dependent packages; do not reuse it
without recording those details.

## Benchmark Classes

The repository contains distinct result classes which must not be compared as a
single benchmark:

| Result class | Source | Appropriate claim |
| --- | --- | --- |
| Hybrid and classical results | `experiments/results/*.json` | Recorded experiment summaries; the dataset version and split are not yet captured in the repository. |
| Supervised pump benchmark | `docs/BENCHMARK_RESULTS.md` | Historical MIMII pump result. |
| Unsupervised DCASE-shaped evaluation | `scripts/evaluate_dc2020.py`, `docs/DC2020_RESULTS.md` | Synthetic pipeline demonstration only; not an official DCASE result. |

## Recording a Real Experiment

For every real-data run, create a dedicated directory under `experiments/` and
store:

1. Dataset name, immutable version or checksum, license, and exact split.
2. Command line, configuration file, and random seed.
3. `requirements.lock` generated from the clean environment.
4. Metrics, model parameters, and model checksum.

Training through `scripts/train.py` accepts `--random-state`; preserve this
value in the experiment metadata. Avoid relying on Python's `hash()` for seeds,
because its output varies between processes.

## Synthetic Regression Run

The DCASE-shaped synthetic runner is deterministic and emits both a CSV and
metadata JSON file:

```bash
python scripts/evaluate_dc2020.py \
  --seed 42 \
  --output experiments/synthetic/metrics.csv
```

Changing the seed deliberately changes the generated data. The resulting metric
values are suitable for regression testing, but are not evidence of performance
on DCASE audio or a production workload.
