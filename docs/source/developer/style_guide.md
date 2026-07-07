# Documentation Style Guide

This guide defines the writing and structure standards for Res-IRF4 documentation.

## Scope

Apply this guide to all files in `docs/source/`.

## Audience-first writing

- State who the page is for in the first paragraph.
- Put the expected outcome near the top.
- Use imperative voice for procedural steps.
- Keep domain assumptions explicit (for example, baseline year, policy scope, scenario family).

## Page template

Use this order for most pages:

1. one-line purpose statement
2. prerequisites (if any)
3. procedure or conceptual content
4. expected outputs or checks
5. troubleshooting and related links

## Headings and naming

- Use short, descriptive headings.
- Prefer sentence case for section names.
- Keep filenames lowercase with underscores.
- Avoid ambiguous titles like `notes.md` or `misc.md`.

## Commands and paths

- Wrap commands in fenced `bash` blocks.
- Keep commands copy-paste ready.
- Use workspace-relative paths (for example `project/config/config.json`).
- When documenting flags, include defaults when known.

## Tables, figures, and references

- Add a short caption or context sentence before each figure/table.
- Prefer links to canonical pages instead of duplicating long explanations.
- Keep bibliographic references in `articles.bib` where applicable.

## Legacy handling

- v4 content is primary.
- v3.0/2012 material must remain in the Legacy Archive and be clearly labeled as legacy.
- Do not place new v4 guidance inside legacy pages.

## Quality checklist for page authors

Before opening a PR:

1. build docs locally with strict warnings (`sphinx-build -b html -W --keep-going source _build/html`)
2. check for broken internal links
3. confirm page appears in a `toctree`
4. verify command snippets and paths against current repo layout
5. add or update related page links (`seealso` or inline links)
