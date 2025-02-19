# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.1] - 2024-02-09

### Added

- All numpy functions now use only one thread, to avoid clash with PIPE multithreading on a higher level. For some platforms this improved performance by a factor 2.
- Added check for the presence of calibration files

## [1.0.0] - 2024-02-06

### Added

- Initial release
- CHANGELOG.md created
