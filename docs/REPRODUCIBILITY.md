# Reproducibility

## Clean environment

The primary full-pipeline installation may use an existing, verified SimNIBS
4.6.0 environment:

```bash
conda activate simnibs
python -m pip install --no-deps dwi2cond_xp-0.2.0-py3-none-any.whl
python -c "import simnibs, dwi2cond_xp; assert simnibs.__version__ == '4.6.0'"
```

To create an independent environment:

```bash
conda env create -f environment.yml
conda activate dwi2cond-xp-simnibs46
python -m pip install -e '.[test]'
python -m pip check
ruff check src tests tools scripts
pytest -q
python -m build
```

Then install the wheel into a fresh Python 3.11 environment and run:

```bash
dwi2cond-xp --help
dwi2cond-xp fit-dti --help
dwi2cond-xp register-tensor --help
dwi2cond-xp simulate-tdcs --help
```

## Reference tools

FSL is never downloaded by CI. Optional FSL numerical tests require the caller
to configure a local FSL 6.0.4 installation and must report a visible skip when
it is absent. SimNIBS integration tests require exactly version 4.6.0.

## Real data

Real-data reproduction requires the user to obtain diffusion and anatomical
data under their original terms, select a supported `dwi2cond-xp` preprocessing
branch, and create a SimNIBS 4.6 CHARM head model. Inputs outside the fixed
SimNIBS 4.6 subset, such as wrapped phase requiring PRELUDE, must be prepared by
an appropriate external tool. Do not upload identifiable or controlled-access
medical images to an issue.

Record software versions, DWI shape and shell, masks, tensor ordering, world
transform, interpolation order, conductivity mode, solver, CPU count, output
hashes, and QA JSON for every scientific reproduction.
