# Architecture

`core/` contains reusable graph/ML code.

`experiments/` contains research experiments and benchmark entry points; these are not imported by the live API.

`api/` exposes graph inference.

`scripts/` contains only the small set of commands a presenter should need.

This separation keeps exploratory experiment files out of the root while preserving the existing research code.
