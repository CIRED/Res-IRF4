# Contributing

## Documentation contribution workflow

Documentation source lives in `docs/source/` and is built with [Sphinx](https://www.sphinx-doc.org/).
Published docs: <https://cired.github.io/Res-IRF4/>

## Local docs setup

```bash
pip install -r docs/requirements.txt
```

## Local quality checks

Run from repository root:

```bash
cd docs
make clean
sphinx-build -b html -W --keep-going source _build/html
sphinx-build -b linkcheck -W --keep-going source _build/linkcheck
```

Open `_build/html/index.html` in your browser to preview.

## Adding or editing pages

- Write docs pages as `.md` or `.rst` under `docs/source/`.
- Add each new page to an appropriate `toctree` (usually in `docs/source/index.rst`).
- Store images in `docs/source/img/` and tabular assets in `docs/source/table/`.
- Follow the [Documentation Style Guide](style_guide.md).

## Docs PR checklist

1. page is linked in navigation (`toctree`)
2. commands and paths were tested
3. local Sphinx build passes with warnings as errors
4. local linkcheck passes
5. legacy references are clearly labeled as legacy where relevant

## Git workflow

Create a branch:

```bash
git checkout master
git pull
git checkout -b <branch_name>
```

Update from `master`:

```bash
git checkout master
git pull
git checkout <branch_name>
git merge master
```

Push branch:

```bash
git push origin <branch_name>
```
