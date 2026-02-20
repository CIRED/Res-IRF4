# Installation Help

Use this page when the normal [Quickstart](quickstart.md) flow fails.

## Conda environment checks

Create and activate:

```bash
conda env create -f environment.yml
conda activate Res-IRF4
```

Verify:

```bash
conda env list
python --version
```

Expected Python version is `3.8.x`.

## Common setup issues

### Environment already exists

```bash
conda env remove -n Res-IRF4
conda env create -f environment.yml
```

### Wrong environment activated

```bash
conda activate Res-IRF4
which python
```

The reported interpreter should be inside your conda environment path.

### Package resolution fails

Try cleaning stale solver metadata:

```bash
conda clean --all -y
conda env create -f environment.yml
```

## Jupyter kernel setup (optional)

If you use notebooks:

```bash
conda activate Res-IRF4
python -m ipykernel install --user --name Res-IRF4 --display-name "Python (Res-IRF4)"
```

Useful commands:

```bash
jupyter kernelspec list
jupyter kernelspec remove Res-IRF4
```

## Reproducible environment export

To share your exact local environment state:

```bash
conda activate Res-IRF4
conda env export > environment.lock.yml
```
