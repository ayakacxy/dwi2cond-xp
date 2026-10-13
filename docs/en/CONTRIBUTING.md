# Contributing

[English](../en/CONTRIBUTING.md) | [简体中文](../zh/CONTRIBUTING.md)

[Documentation](README.md) · [Project home](../../README.md)

Contributions are welcome through focused issues and pull requests.

1. Do not include human MRI data, subject identifiers, credentials, local
   absolute paths, or generated FEM outputs.
2. Keep the pure-Python reference path. New acceleration must be explicit and
   must not silently replace or fall back from a requested backend.
3. Do not reduce resolution, data, iterations, or scientific checks to obtain a
   performance result.
4. Write new comments and docstrings in Chinese. Keep CLI help and error
   messages in English.
5. Run `ruff check src tests tools scripts` and `pytest -q`.
6. Report numerical error and timing boundaries for reference comparisons.

Large changes should begin with an issue that describes the scientific
requirements, input data that can legally be shared, and the proposed validation.
