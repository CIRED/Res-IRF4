# Contributing

## Updating Documentation

### Overview

Documentation source lives in `docs/source/` and is built with [Sphinx](https://www.sphinx-doc.org/).
It is deployed automatically to GitHub Pages via GitHub Actions on every push to `master`.


Live documentation: <https://cired.github.io/Res-IRF4/>

### Local build

1. Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

2. Build HTML locally:

```bash
cd docs
make clean
make html
```

3. Open `docs/_build/html/index.html` in your browser to preview.

### Adding content

- Documentation pages are Markdown (`.md`) or reStructuredText (`.rst`) files in `docs/source/`.
- Add new pages to the `toctree` in `docs/source/index.rst`.
- Images go in `docs/source/img/`, data tables in `docs/source/table/`.

## Git workflow

### Creating a branch

1. Update your local copy of master:

```bash
git pull
```

2. Create and switch to a new branch:

```bash
git checkout -b name_of_new_branch
```

3. Push the branch to GitHub:

```bash
git push origin name_of_new_branch
```

### Updating a branch from master

```bash
git checkout master
git pull
git checkout name_of_the_branch
git merge master
```
