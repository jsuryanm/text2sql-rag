# Fix: `ModuleNotFoundError: No module named 'pdfminer.psexceptions'`

## Symptom

Running the PDF parsing path (`unstructured.partition.auto.partition` on a
`.pdf` file, e.g. via `scripts/parser.py`) failed with:

```
ModuleNotFoundError: No module named 'pdfminer.psexceptions'
```

The traceback showed the error originating inside `unstructured`'s own code,
not in this project's code:

```
unstructured/partition/pdf_image/pdfminer_utils.py:9
    from pdfminer.psexceptions import PSSyntaxError
```

## Root Cause

`pyproject.toml` pinned an old version of `pdfminer.six`:

```toml
"pdfminer-six==20231228",  # Pin compatible version for unstructured
```

That pin was added to satisfy `unstructured`, but `unstructured` itself has no
version ceiling on `pdfminer.six` (`pyproject.toml`/wheel metadata just says
`pdfminer.six`, no version range). Over time the installed `unstructured`
package moved forward to `0.18.32`, whose internal code now imports
`pdfminer.psexceptions` — a module that was only added to `pdfminer.six` in a
release **after** `20231228` (Dec 2023).

So the versions had drifted apart:

- `unstructured==0.18.32` (new) → expects the new `pdfminer.six` module layout
- `pdfminer-six==20231228` (old, pinned) → doesn't have that module yet

Installing the old pin against the new `unstructured` produced the
`ModuleNotFoundError`.

(Along the way, the local `.venv` install of `pdfminer.six` was also found to
be physically incomplete — only 7 of the ~178 files the package should ship
were present. `pip show`/`pip list` reported the package as installed and
`pdfminer.six-...dist-info` existed, but most of the actual `.py` files were
missing. This was cleaned up with `uv pip install --force-reinstall --no-cache pdfminer-six` before diagnosing the real version-mismatch issue.)

## Fix

Bump the `pdfminer-six` pin to a version that ships `pdfminer/psexceptions.py`:

```toml
"pdfminer-six>=20240706",  # Pin compatible version for unstructured (needs pdfminer.psexceptions)
```

Then re-lock and re-sync the environment:

```bash
uv lock
uv sync
```

`uv lock` resolved this to the newest available release (`20260107` at the
time of the fix); `uv sync` installed it into `.venv`.

## Verification

```bash
uv run python -c "from pdfminer.psexceptions import PSSyntaxError; print('PDFMiner OK')"
uv run python scripts/parser.py
```

Both ran cleanly — the import succeeded and `scripts/parser.py` produced the
parsed `unstructured` elements list for `data/transformers_paper.pdf` with no
errors.

## Takeaway

When a dependency (`unstructured`) declares another dependency
(`pdfminer.six`) with **no version bound**, pinning that transitive
dependency to a fixed old version can silently break as the first dependency
is upgraded — the pin was written for an older `unstructured` release and
never revisited. If this happens again with another transitive dependency,
check whether the pin still matches what the current version of the
depending package actually expects, rather than assuming the old pin is
still correct.
