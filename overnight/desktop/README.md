# Desktop worker

The laptop assigns work by appending to `tasks.json` on `main`; the desktop runs
`bash overnight/desktop_worker.sh` from its clone and commits one `results/<name>.txt`
plus the full gauntlet log per task back to `main`. A task with a result file is done.

Task fields: `name` (challenger dir), `sed` (switch to flip in agent.py; empty = the
tree as is), `kind` (`switch` = gauntlet, `clocktest`), `champion` (default `.`, the
checkout), `games`, `openings` (`default` or `platform`), `workers`, `elo0`, `elo1`,
`base_ms` (8000 or 120000).
