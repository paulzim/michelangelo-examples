**What type of PR is this? (check all applicable)**
- [ ] Refactor
- [ ] Feature
- [ ] Bug Fix
- [ ] Optimization
- [ ] Documentation Update

> **Commit message format matters here.** `CHANGELOG.md` and GitHub Release notes are generated automatically from commit messages via [git-cliff](https://git-cliff.org/) — there's no manual changelog step. Use [Conventional Commits](https://www.conventionalcommits.org/) (`type(scope): description`, e.g. `feat(california-housing): add xgb_train pipeline`) so this PR's commit(s) land in the right changelog section. See the `pr-template` skill for the full type/scope reference.

<!-- Describe what has changed in this PR -->
**What changed?**


<!-- Tell your future self why have you made these changes -->
**Why?**


<!-- How have you verified this change? Ran it locally? Built the Docker image? Ran it through a Michelangelo sandbox? -->
**How did you test it?**


<!-- Assuming the worst case, what can be broken for someone installing/running this example? -->
**Potential risks**

<!-- Does this PR introduce a breaking change? Check all that apply. -->
**Breaking Changes**

- [ ] No breaking changes
- [ ] API changes (public functions or classes under `src/michelangelo_examples/`)
- [ ] Dependency changes (new/changed `michelangelo` version pin, new required extra)
- [ ] Docker/image changes (base image, exposed entrypoint behavior, image tag)
- [ ] Pipeline/project config changes (`pipeline.yaml`/`project.yaml` schema, required env vars)

> If any breaking change box is checked (other than "No breaking changes"), you **must**:
> 1. Use the `BREAKING CHANGE:` footer **or** the `!` suffix (e.g. `feat!:`) in your commit message.
> 2. Fill in the **Migration guide** section below with step-by-step upgrade instructions.

<!-- Required only if a breaking change box above is checked. Delete this section if no breaking changes. -->
**Migration guide**

_Describe the steps a user must take to upgrade. Include before/after examples for any API, config, or dependency change._

<!-- CHANGELOG.md and the GitHub Release body are generated automatically from commit messages via git-cliff -- nothing to fill in here manually. Just confirm your commit message's type/scope/description accurately describes the change, since that text becomes the changelog entry verbatim. -->
**Release notes**

_Auto-generated from commit messages via git-cliff. Confirm above that your commit message follows Conventional Commits and reads well standalone (it will appear in `CHANGELOG.md` as-is)._

<!-- Does this PR introduce a user-facing change? Is the relevant README (root or per-project/per-pipeline) updated? -->
**Documentation Changes**
