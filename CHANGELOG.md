# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Resolved CI/CD pipeline failures by removing unavailable MCP dependency
- Fixed Python version compatibility (correctly set to >=3.8)
- Added missing runtime dependencies (psutil, aiohttp, jinja2, reportlab, schedule, rich)
- Added missing type stubs for mypy (types-PyYAML, types-aiofiles, types-redis, etc.)
- Created mock MCP module for testing when MCP package is not available
- Fixed import errors and improved error handling

### Changed
- Updated dependency management to use pydantic>=2.0.0
- Improved code formatting and removed unused imports
- Cleaned up temporary files and cache directories
- Enhanced documentation and installation instructions

## [1.0.0] - 2024-06-29

### Added
- Initial release with 39 graph analysis tools
- Complete NetworkX integration via MCP
- Redis persistence support
- Multi-format visualization (matplotlib, Plotly, pyvis)
- Advanced algorithms including community detection and ML integration
- Comprehensive security features
- Production-ready monitoring and audit logging
- Docker support
- Extensive test coverage

### Security
- Input validation for all operations
- Path traversal protection
- Resource usage limits
- Secure temporary file handling

### Performance
- Optimized for graphs up to 10,000 nodes
- Streaming support for large graph operations
- Caching for expensive computations
