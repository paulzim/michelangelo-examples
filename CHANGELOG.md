# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-08-11


### Bug Fixes


- **ci:** Changelog.yml PR creation fails on real tag pushes (detached HEAD) (#14)



### CI/CD


- Add Dependabot config for weekly grouped dependency updates (#18)



### Miscellaneous


- Stop hardcoding model_name/report_name in push steps (#15)


- Bump michelangelo floor to 0.8.0 (#26)


- Bump michelangelo-examples version to 0.3.0 (#28)



## [0.2.0] - 2026-07-28


### Bug Fixes


- **xgb_train:** Re-qualify Ray's scheme-less checkpoint path before fsspec lookup (#11)



### CI/CD


- **release:** Generate CHANGELOG.md and GitHub Release notes via git-cliff (#7)


- Publish multi-arch (amd64+arm64) examples image (#9)



### Documentation


- **pr-template:** Make PR template and skill cliff-friendly (#6)


- Update CHANGELOG.md for v0.1.0 (#8)



### Miscellaneous


- Bump michelangelo floor to 0.6.0 (#12)


- Release 0.2.0 (#13)



## [0.1.0] - 2026-07-22


### Bug Fixes


- **california-housing:** Bump michelangelo to 0.4.0rc2, add minio dep (#2)



### Features


- **california-housing:** Add pytorch_lightning_train pipeline (#1)


- **california-housing:** Add xgb_train pipeline (port from core michelangelo) (#4)


- Publish to PyPI, keep michelangelo pin in sync (#5)



### Miscellaneous


- Repo skeleton and package scaffold


- Add examples/config/project.yaml



### Refactoring


- **california-housing:** Rename pytorch_lightning_train pipeline to pytorch_train (#3)




