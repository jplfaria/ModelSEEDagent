# ModelSEEDagent – Development Roadmap (June 2025)

> This concise roadmap supersedes older planning files that have been moved to `docs/archive/`.  It only tracks verifiable facts and the **next concrete steps**.

## Current Working State

| Area | Status |
|------|--------|
| Interactive interface | Stable & production-ready (`python -m src.interactive.interactive_cli`) |
| LLM back-ends | Argo, OpenAI & Local verified |
| Test suite | 43 passed · 4 skipped · 0 failed |
| Tool coverage | 23 / 25 tools exercised in validation suite |
| Documentation | README & User Guide accurate |
| Website | MkDocs site builds & deploys |

## Known Issues

1. **Typer CLI import errors** – some sub-modules still reference old paths.
2. **`modelseed-agent setup`** – option parsing & prompts need clean-up.
3. **Help formatting** – long descriptions overflow typical 80-col terminals.
4. **Async tests** – 4 tests still marked `skip`; need proper `pytest.mark.asyncio`.

## Phase 1 – Stability Fixes (in progress)

- [ ] Resolve remaining relative-import problems in `src/agents/` and `src/cli/`.
- [ ] Finalise entry-point wiring in `pyproject.toml`.
- [ ] Re-enable the 4 skipped async tests → target ≥ 95 % pass rate.
- [ ] Tidy Typer help width & styling.

## Phase 2 – CLI Polish (queued)

- [ ] Harmonise option names across commands (`--backend` vs `--llm-backend`).
- [ ] Add `modelseed-agent doctor` environment diagnostic command.
- [ ] Smoke-test matrix: Linux/macOS, Python 3.9-3.12.

## Phase 3 – Documentation & Website

- [ ] Auto-publish changelog snippets to GitHub Releases.
- [ ] Embed short tutorial videos in the User Guide.
- [ ] Surface validation dashboards inside docs site (iframe embed).

---

_Last updated: 2025-06-16_
