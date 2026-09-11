# Learning pytest with this codebase

This guide teaches pytest from zero, using the real tests written for
`app/services/*.py` in this repo as examples. Read it top to bottom once,
then use it as a reference while you read the actual test files in
`tests/units/`.

Everything referenced here already exists in the repo and passes:

```bash
uv run pytest tests/units -v
```

---

## 1. What is pytest, and why bother?

Every function you write makes assumptions: "if the file doesn't exist,
raise `FileNotFoundError`", "if the cache is empty, return `None`". A test
is just code that calls your function and checks the assumption held.

Why not just try it manually in a Python shell? Because:
- You'd have to re-type the same manual check every time you change the code.
- pytest can run hundreds of checks in seconds and tell you exactly which one broke.
- A failing test is proof something is wrong *before* a user hits it.

This project actually had several real bugs (inverted `if` conditions,
missing `return` statements, a missing `await`) that were only found by
reading the code carefully — the kind of thing a test would have caught
immediately. Section 7 walks through each one.

---

## 2. Anatomy of a test

Open [tests/units/test_document_service.py](../tests/units/test_document_service.py). A minimal test looks like this:

```python
def test_reads_txt_file_content(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Hello, this is a test document.")

    text = parse_document(str(file_path))

    assert text == "Hello, this is a test document."
```

Three parts, always in this order (sometimes called **Arrange / Act / Assert**):
1. **Arrange** — set up the input (`file_path`, its content).
2. **Act** — call the real function you're testing (`parse_document`).
3. **Assert** — check the result is what you expected.

`assert <expression>` is the whole mechanism. If the expression is truthy,
the test passes silently. If it's falsy, pytest raises `AssertionError` and
prints a detailed diff of what the two sides actually were.

### How pytest finds your tests

pytest doesn't need a registry — it just looks for a pattern. Per this
project's `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

- Any file named `test_*.py` under `tests/` is collected.
- Any function named `test_*` inside it becomes a test.
- Any class named `Test*` inside it is scanned for `test_*` methods too
  (this is just for organizing related tests — plain functions work fine
  without a class).

### Running tests

```bash
uv run pytest                                  # everything
uv run pytest tests/units -v                   # just the unit tests, verbose
uv run pytest tests/units/test_cache_service.py             # one file
uv run pytest tests/units/test_cache_service.py::TestClearCache::test_clear_all_documents  # one test
uv run pytest -k "cache"                        # anything with "cache" in the name
uv run pytest -m "not slow"                     # skip tests marked @pytest.mark.slow
```

`-v` (verbose) prints one line per test instead of just dots. `-k` is a
quick filter by name substring — handy while you're only working on one
piece.

---

## 3. Fixtures: reusable setup

Copy-pasting the same setup code into every test gets old fast. A
**fixture** is a function decorated with `@pytest.fixture` that produces
some piece of test setup; any test function that names it as a parameter
gets the result automatically — pytest wires it up for you.

### Built-in fixture: `tmp_path`

`tmp_path` is provided by pytest itself. It's a fresh, empty directory
(a `pathlib.Path`) that exists only for the duration of that one test, then
gets cleaned up. That's how `test_reads_txt_file_content` above creates a
real file on disk without leaving junk behind or clashing with other tests.

### Custom fixtures: `tests/conftest.py`

`conftest.py` is a special filename — pytest automatically loads it before
running any tests in that directory (and subdirectories), and every fixture
defined in it becomes available to every test file nearby, with no import
needed. This project's shared fixtures live in
[tests/conftest.py](../tests/conftest.py):

```python
@pytest.fixture
def sample_chunks():
    """A tiny, realistic list of document chunks."""
    return [
        {"text": "This is the first chunk of text.", "metadata": {"page": 1, "tokens": 7}},
        {"text": "This is the second chunk of text.", "metadata": {"page": 1, "tokens": 7}},
    ]
```

Any test can now just ask for `sample_chunks` as a parameter:

```python
def test_save_and_load_chunks(self, local_storage, sample_chunks):
    local_storage.save_chunks(doc_id, file_extension, sample_chunks)
    ...
```

pytest sees the parameter name `sample_chunks`, finds the fixture with that
name, calls it, and passes the return value in. You never call the fixture
function yourself.

### Fixtures that depend on other fixtures

Fixtures can request other fixtures, the same way. In
[tests/units/test_storage_backends.py](../tests/units/test_storage_backends.py):

```python
class TestLocalStorageBackend:
    @pytest.fixture
    def local_storage(self, tmp_path):
        """Create a LocalStorageBackend with temporary directory."""
        return LocalStorageBackend(cache_dir=tmp_path)
```

`local_storage` depends on `tmp_path`. This fixture is defined *inside* the
test class, so it's only available to tests in that class — that's fixture
*scoping*: fixtures in `conftest.py` are global, fixtures inside a class are
local to it. Use class-scoped fixtures when the setup only makes sense for
one group of tests.

---

## 4. Mocking: testing without the real thing

Some of the code in this project talks to external systems: AWS S3, OpenAI,
Redis. We don't want tests that:
- Cost real money (OpenAI API calls)
- Need real credentials and network access to even run
- Could accidentally delete real data
- Are slow and flaky compared to pure Python

The fix is to replace ("mock") the external dependency with a fake that
behaves the same way, just in memory. This project uses three different
techniques depending on the situation — pick the simplest one that works.

### Technique 1: hand-written fake classes

For `CacheService`, instead of mocking individual method calls, we just
write a tiny real class that implements the same interface. See
[tests/units/test_cache_service.py](../tests/units/test_cache_service.py):

```python
class FakeStorageBackend(StorageBackend):
    """An in-memory stand-in for LocalStorageBackend/S3StorageBackend."""

    def __init__(self):
        self.documents = {}

    def exists(self, document_id, file_extension):
        doc = self.documents.get(document_id)
        return doc is not None and all(k in doc for k in ("chunks", "embeddings", "metadata"))
    ...
```

Then we just construct `CacheService` with it:

```python
cache_service = CacheService(storage_backend=FakeStorageBackend())
```

This works because `CacheService` was written to accept *any*
`StorageBackend`, not hardcoded to a specific one — that's the whole point
of `storage_backend.py` being an abstract interface (see Section 6). This
approach is preferred here over `unittest.mock.MagicMock` because the fake
class is explicit: you can read exactly what it does, step through it in a
debugger, and it can't accidentally accept a typo'd method name the way a
`MagicMock` silently would.

The same idea is used for Redis in
[tests/units/test_query_cache_service.py](../tests/units/test_query_cache_service.py) (`FakeRedisClient`)
and for OpenAI in [tests/units/test_embeddings_service.py](../tests/units/test_embeddings_service.py) (`FakeEmbeddingsClient`).

### Technique 2: `monkeypatch`

`monkeypatch` is a built-in pytest fixture for temporarily replacing an
attribute, and it automatically undoes the change after the test finishes
— even if the test fails. Used in
[tests/units/test_docling_service.py](../tests/units/test_docling_service.py) to swap out
Docling's real `DocumentConverter` for a fake:

```python
def test_convert_document_success(monkeypatch, tmp_path):
    class FakeConverter:
        def convert(self, path):
            return fake_result

    monkeypatch.setattr(module, "DocumentConverter", FakeConverter)

    result = module.convert_document(str(file_path))
```

`monkeypatch.setattr(target, "name", replacement)` replaces
`target.name` with `replacement` for the duration of this one test only.
It's also used for environment variables: `monkeypatch.setenv(...)` (see
the S3 test below), and to swap out one attribute on a real object, like
`monkeypatch.setattr(settings, "OPENAI_API_KEY", None)` in
`test_embeddings_service.py`.

### Technique 3: `moto` (fake AWS)

`S3StorageBackend` talks to real AWS S3. `moto` is a library that
intercepts every `boto3` call and answers them from an in-memory fake S3,
so the code under test never knows the difference. See
[tests/units/test_s3_storage.py](../tests/units/test_s3_storage.py):

```python
from moto import mock_aws

@pytest.fixture
def s3_storage(aws_credentials):
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET_NAME)
        yield S3StorageBackend(bucket_name=BUCKET_NAME)
```

Everything inside the `with mock_aws():` block — bucket creation, and later
whatever the test does through `s3_storage` — runs against the fake S3.
boto3 still insists on *looking like* it has AWS credentials even though
moto never actually contacts AWS, so a small fixture sets fake ones via
`monkeypatch.setenv(...)` first.

### When to use which

- Testing your own class that already accepts a pluggable dependency
  (like `StorageBackend`) → write a small fake class (Technique 1).
- Swapping one function/attribute on an existing object for one test →
  `monkeypatch` (Technique 2).
- Testing code that hardcodes a real cloud SDK (`boto3`) → use the SDK's
  matching fake library if one exists, like `moto` for AWS (Technique 3).

---

## 5. Testing async code

`EmbeddingService.generate_embeddings()` is declared `async def` because it
awaits a network call. You can't just call an async function directly and
get a result — you have to run it inside an event loop.

`pytest-asyncio` handles this. This project has it configured with:

```toml
asyncio_mode = "auto"
```

in `pyproject.toml`, which means **any** `async def test_...` function is
automatically run as an async test — no decorator needed. From
[tests/units/test_embeddings_service.py](../tests/units/test_embeddings_service.py):

```python
async def test_empty_list_returns_empty(self, embedding_service):
    embeddings, usage = await embedding_service.generate_embeddings([])
    assert embeddings == []
    assert usage is None
```

Note the `async def` on the test itself, and `await` when calling the async
function under test — same as you'd write in real application code.

(Without `asyncio_mode = "auto"`, you'd need to add
`@pytest.mark.asyncio` above every async test function manually.)

---

## 6. Markers: grouping and skipping tests

This project defines four custom markers in `pyproject.toml`:

```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "aws: marks tests that require AWS services",
]
```

You apply one with a decorator:

```python
@pytest.mark.slow
def test_something_that_takes_ten_seconds():
    ...
```

Then select or deselect by marker with `-m`:

```bash
uv run pytest -m "not slow"        # skip anything marked slow
uv run pytest -m "aws"             # only run AWS-related tests
```

None of the new tests in this guide are marked `slow` — they all run in
well under a second each, since everything external is faked.

---

## 7. What each test file covers (and the bugs it would have caught)

All test files below live in `tests/units/`.

### `test_storage_backend.py` — the abstract interface

`StorageBackend` is an `ABC` (Abstract Base Class): it lists the methods
every storage backend *must* implement, but has no logic of its own. The
tests just confirm Python enforces that contract — you can't instantiate
`StorageBackend` directly, and a subclass missing even one method also
can't be instantiated.

### `test_storage_backends.py` — `LocalStorageBackend`

Save/load round-trips for chunks, embeddings, metadata, and the original
document, all against a real (but temporary, via `tmp_path`) filesystem
directory — no mocking needed since local disk *is* the thing being tested.

**Bug this would have caught:** `load_chunks()` computed the chunks data
but never had a `return` statement, so it always returned `None`. Fixed by
adding `return chunks` at the end of the function.

### `test_s3_storage.py` — `S3StorageBackend`

Same save/load/exists/delete/list/stats coverage as local storage, but
against a `moto`-faked S3 bucket.

**Bugs this would have caught:**
- `exists()` checked for a file named `documents.{ext}` (plural) but
  `save_document()` actually saves `document.{ext}` (singular) — so
  `exists()` could never return `True` even with everything saved. Fixed
  the typo.
- `load_embeddings()` loaded the array from S3 but never returned it. Fixed
  by adding `return embeddings`.

### `test_cache_service.py` — `CacheService`

Uses a hand-written `FakeStorageBackend` (see Section 4) instead of real
storage, so it tests `CacheService`'s own logic in isolation: SHA-256 based
document IDs, chunk/embedding length validation, and cache clearing.

**Bugs this would have caught:**
- `load_chunks_and_embeddings()` had its `if` condition backwards
  (`if self.cache_exists(...): return None`), so it returned `None`
  whenever the cache *did* exist — exactly backwards. Fixed to
  `if not self.cache_exists(...): return None`.
- `clear_cache()` with no `doc_id` (meaning "clear everything") was a
  no-op that always reported `cleared: False`. Implemented the actual loop
  over `storage.list_documents()` + `storage.delete(...)`.

### `test_query_cache_service.py` — `QueryCacheService`

Covers the "disabled" pass-through mode (no Redis credentials configured —
the app should still work, just without caching), and the "enabled" mode
using a hand-written `FakeRedisClient`.

**Bugs this would have caught:**
- `get()`'s exception handler called `self._record_miss` without `()` —
  referencing the method object instead of calling it, so the miss was
  silently never recorded. Fixed by adding the `()`.
- `delete()` had its enabled-check backwards
  (`if self.enabled: return 0`), so it returned immediately without
  deleting anything whenever the cache *was* enabled. Fixed to
  `if not self.enabled: return 0`.

### `test_embeddings_service.py` — `EmbeddingService`

Uses a `FakeEmbeddingsClient` in place of `langchain_openai.OpenAIEmbeddings`
so no real OpenAI call happens, and `pytest-asyncio` (Section 5) since the
methods are `async def`.

**Bugs this would have caught:**
- The constructor discarded whatever `query_cache_service` you passed in
  and always built a fresh default one instead
  (`if query_cache_service is not None: self.query_cache_service = QueryCacheService()`).
  Also, passing `None` left `self.query_cache_service` unset entirely,
  which would crash the first time it was used. Fixed to actually keep the
  passed-in instance, defaulting only when `None`.
- `generate_embeddings()`'s `return embeddings, usage_info` was indented
  one level too far — inside the `for` loop that fills in cache-miss
  results — so with more than one cache-miss text, it returned after
  processing only the first one. Fixed by dedenting the return to run
  after the loop completes. `test_mixed_cache_hits_and_misses_returns_all_texts`
  is a direct regression test for this.
- `generate_single_embedding()` called `self.generate_embeddings([text])`
  without `await`, so it would try to unpack a coroutine object as a
  2-tuple and crash. Fixed by adding `await`.

### `test_document_service.py` — `parse_document`, `chunk_text`, etc.

Plain function tests against real small `.txt` files in `tmp_path` — the
fast native-read path, so no heavy `unstructured` parsing library involved.
No bugs were found in this file during this session.

### `test_docling_service.py` — Docling integration (pre-existing file, extended)

Uses `monkeypatch` to fake Docling's `DocumentConverter` and
`HybridChunker` classes, since the real `docling` library is a heavy
optional dependency.

**Bugs this would have caught:**
- `_extract_page_numbers()` did `getattr(item, 'prov', None) or None` —
  if `item.prov` was falsy, this evaluated to `None`, and the very next
  line tried to `for prov in None`, which crashes with `TypeError`. Fixed
  to `or []` so the loop is always over something iterable. It also read
  `getattr(prov, 'prov', None)` (looking up a `.prov` attribute on the
  `prov` object itself) instead of the actual page number field,
  `getattr(prov, 'page_no', None)` — fixed to use the correct attribute
  name, and updated the existing test's fake objects (`fake_item`) to
  match.
- `chunk_with_hybrid()` built each chunk's metadata dict and appended it to
  the result list, but that whole block was nested *inside*
  `if chunk.meta and chunk.meta.headings:` — so any chunk without headings
  was silently dropped from the output entirely, not just missing its
  heading metadata. Fixed by dedenting the block so every chunk is kept;
  only the `headings` list itself stays conditional on having headings.

---

## 8. Coverage: how much of the code do the tests actually exercise?

This project runs coverage automatically via `pytest-cov` (configured in
`pyproject.toml`'s `addopts`). Every `pytest` run prints a table like:

```
Name                                  Stmts   Miss  Cover   Missing
-------------------------------------------------------------------
app/services/local_storage.py           100      5    95%   169, 193, 216, 240, 250
```

- **Stmts** — total executable lines in that file.
- **Miss** — lines that *no* test ever executed.
- **Cover** — percentage covered.
- **Missing** — the exact line numbers never hit, so you know precisely
  what to test next.

A full interactive report is also written to `htmlcov/index.html` — open it
in a browser to click through each file and see covered (green) vs.
uncovered (red) lines highlighted directly in the source. High coverage
doesn't guarantee correct code (a test can execute a line without checking
its result is right), but low coverage on a file is a reliable signal that
it's undertested.

```bash
uv run pytest --cov=app --cov-report=html
open htmlcov/index.html   # or just open the file in a browser manually
```

---

## Where to go from here

- Read through `tests/units/test_cache_service.py` end to end — it's the
  most self-contained example combining fixtures, a hand-written fake, and
  clear Arrange/Act/Assert structure.
- Try breaking something on purpose (e.g. re-introduce one of the bugs from
  Section 7) and watch the corresponding test fail — that failure message
  is exactly what you'd see if a real regression slipped in.
- Add a new test for a scenario not covered yet (check the "Missing" column
  in the coverage table for ideas).
