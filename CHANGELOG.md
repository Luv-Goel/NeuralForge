# Changelog

All notable changes to NeuralForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dependabot configuration for automatic dependency updates (pip + GitHub Actions)
- `.editorconfig` for consistent coding style across editors
- Issue templates (bug report, feature request, config)
- Pull request template
- Data augmentation utilities (`neuralforge.utils.augmentations`) with Cutout
- This changelog!

### Fixed
- Hardcoded `.cuda()` calls in DARTS search network that broke CI on CPU-only runners
- Removed ~50 lines of dead/incomplete code from `DiscreteCell.__init__` in genotypes.py
- Fixed missing `neuralforge.utils.augmentations` module import in darts.py

### Changed
- Updated LICENSE year from 2024 to 2026
- Expanded documentation pages (installation, API reference, contributing)
- Cleaned up unused variable declarations in genotype module

## [0.2.0] - 2024-01-01

### Added
- DARTS (Differentiable Architecture Search) algorithm
- Regularized Evolution search algorithm
- Random search baseline
- Cell-based search space with configurable primitives
- Genotype encoding/decoding with mutation and crossover
- Model profiling (parameters, MACs, FLOPs, memory)
- Visualization utilities (cell graph rendering, search trajectory plots)
- PyTorch Lightning integration
- Surrogate predictor interface
- Distributed population support
- Comprehensive test suite (36 tests)
- CI/CD pipeline with linting, testing, and type checking
- MIT License, Code of Conduct, Contributing Guide, Security Policy

### Fixed
- Cell operations and auxiliary head stride bugs
- Profiling string formatting issues

## [0.1.0] - 2023-06-01

### Added
- Initial project structure
- Core operations (ReLUConvBN, SepConv, DilConv, etc.)
- Basic search space definition
- Project documentation and README
