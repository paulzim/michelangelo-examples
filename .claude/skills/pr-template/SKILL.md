---
name: pr-template
description: Create or update a GitHub PR using the repo's pull_request_template.md format. Use when creating a new PR or updating an existing PR body.
user-invocable: true
---

Create or update a GitHub PR following the repo's pull request template.

## Template Format

Read the template from `.github/pull_request_template.md` and use it as the exact structure for the PR body.

## Your Task

1. Run `git diff main...HEAD` (or `git diff HEAD~1` if already on main) to understand the changes
2. Infer the PR type(s) from the diff and check the applicable boxes
3. Fill in all sections — keep each answer concise (1-3 sentences or bullet points)
4. **If creating a new PR:**
   - Check if already on a non-main branch; if on main, ask the user for a branch name
   - Push the branch if not yet pushed: `git push -u origin <branch>`
   - Create the PR: `gh pr create --title "<title>" --body "<body>"`
5. **If updating an existing PR:**
   - Get the current PR number: `gh pr view --json number -q '.number'`
   - Update the body: `gh pr edit <number> --body "<body>"`

## PR Title

- Use conventional commit format: `<type>(<scope>): <short description>`
- Types: `feat`, `fix`, `docs`, `ci`, `refactor`, `chore`, `test`, `perf`
- Scopes (optional): a project name (e.g. `california-housing`), `ci`, `docker`
- Keep under 70 characters
- Examples: `feat(california-housing): add xgboost_train pipeline`, `fix(docker): install s3fs for fsspec S3 access`
- **This title becomes the commit message on `main`** (this repo squash-merges), which is what `git-cliff` parses to build `CHANGELOG.md` and the GitHub Release notes on every tag push — get the type/scope/description right here, not just in the PR body's "Release notes" field.

## Changelog / git-cliff mapping

Once merged, `type` determines which `CHANGELOG.md` section the commit lands in:

| Type | Changelog section |
|---|---|
| `feat` | Features |
| `fix` | Bug Fixes |
| `docs` | Documentation |
| `perf` | Performance |
| `refactor` | Refactoring |
| `ci` | CI/CD |
| `test` | Testing |
| `chore` / anything else | Miscellaneous |

For a breaking change, the `BREAKING CHANGE:` footer or `!` suffix (e.g. `feat!:`) is required for it to show up in the changelog's **Breaking Changes** section — checking the PR template's breaking-change boxes alone does nothing to the generated changelog, only the commit message does.

## Filling in the Template

- **What changed?** — What the code does differently now. Be concrete.
- **Why?** — The motivation: bug, new example/project, cleanup, requirement.
- **How did you test it?** — Commands run (`uv sync`, local `python -m` run, `docker build`), or a Michelangelo sandbox run.
- **Potential risks** — Anything that could break for someone installing/running this example. Default to "None" for docs/chore changes.
- **Breaking Changes** — Check applicable boxes: API (public functions/classes), dependency changes (`michelangelo` version pin, new required extra), Docker/image changes, pipeline/project config changes (`pipeline.yaml`/`project.yaml` schema, required env vars). Check "No breaking changes" if none apply. If any breaking change is checked, use `BREAKING CHANGE:` footer or `!` suffix in the commit — see Changelog / git-cliff mapping above.
- **Migration guide** — Required if any breaking change box is checked. Describe step-by-step upgrade instructions with before/after examples. Delete section if no breaking changes.
- **Release notes** — Auto-generated from the commit message via git-cliff, not manually written. Just confirm the commit message (PR title) reads well standalone and accurately describes the change — it will appear in `CHANGELOG.md` verbatim.
- **Documentation Changes** — Note which README(s) were updated (root, per-project, or per-pipeline) or "N/A".
