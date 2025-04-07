# Supplementary material for "Reducing quantum error correction overhead using soft information"

This repository contains Python code for generating synthetic soft measurement samples from quantum error correction circuits, as presented in the paper ["Reducing quantum error correction overhead using soft information"](http://arxiv.org/abs/2504.03504) by Joonas Majaniemi and Elisha S. Matekole.

## Repository structure

- `pyproject.toml` specifies the dependencies used by this software library.
- `data/` contains [STIM](https://github.com/quantumlib/Stim) circuits of quantum memory experiments with superconducting qubits (under `superconducting/`) and neutral atom qubits (under `neutral_atom/`).
- `docs/` contains an interactive Python notebook which shows how to generate synthetic soft measurement samples from a STIM circuit for superconducting and neutral atom qubits.
- `soft_information_models/` contains source code for the project, specifically the measurement probability density functions (PDFs) used to sample soft measurements.
- `tests/` contain unit tests of the measurement PDFs.

## Development environment

We use [poetry](https://python-poetry.org/) to manage the Python dependencies. To get started with the project, make sure to have `poetry` installed and run
```
poetry install --all-extras
```
to install the development dependencies in the appropriate virtual environment. The unit tests to this package can be run via the command
```
poetry run pytest
```

## Citation

Joonas Majaniemi and Elisha S. Matekole, _Reducing quantum error correction overhead using soft information_, (2025) arXiv:2504.03504.

## Notice

(c) Copyright Riverlane 2022-2025. All rights reserved.