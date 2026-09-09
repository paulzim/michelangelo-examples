# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - 2026-09-09


### CI/CD


- Bump softprops/action-gh-release from 2 to 3 (#31)


- Bump peter-evans/create-pull-request from 6 to 8 (#32)


- Bump docker/setup-buildx-action from 3 to 4 (#34)


- Bump docker/build-push-action from 5 to 7 (#46)


- Bump docker/login-action from 3 to 4 (#45)


- Bump docker/metadata-action from 5 to 6 (#47)



### Documentation


- Update CHANGELOG.md for v0.4.0 (#38)



### Features


- **california_housing:** Wire tabular_assembler into the Lightning pipeline (#36)


- **california_housing:** Tar the deployable package before push (#39)


- **california-housing:** Add daily noon trigger to pytorch_train and xgb_train (#43)



### Miscellaneous


- **deps:** Bump pyarrow, pyspark, and s3fs together (#30)


- **deps:** Bump michelangelo from 0.8.0 to 0.9.0 in the uv-minor-and-patch group across 1 directory (#42)


- Bump michelangelo floor to 0.10.0 (#49)


- Release 0.5.0 (#50)



## [0.4.0] - 2026-08-25


### CI/CD


- Bump actions/upload-artifact from 4 to 7 (#20)


- Bump actions/checkout from 4 to 7 (#21)


- Bump astral-sh/setup-uv from 3 to 7 (#22)



### Documentation


- Update CHANGELOG.md for v0.3.0 (#29)



### Miscellaneous


- Populate kind in california-housing pusher config (#16)


- **deps:** Bump pyarrow from 19.0.1 to 25.0.0 (#24)


- Bump michelangelo-examples version to 0.4.0 (#35)



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




