# Release Checklist

This checklist ensures all steps are completed before creating a new release.

## Pre-Release Checklist

### Code Quality
- [ ] All tests pass (`pytest tests/`)
- [ ] Code coverage is above 80% (`pytest --cov`)
- [ ] No linting errors (`ruff check .`)
- [ ] Code is properly formatted (`black --check .`)
- [ ] Type checking passes (`mypy src/networkx_mcp/`)
- [ ] No security vulnerabilities (`bandit -r src/`)
- [ ] Dependencies are up to date (`pip list --outdated`)

### Documentation
- [ ] CHANGELOG.md is updated with new features/fixes
- [ ] API documentation is updated
- [ ] README.md reflects any new features
- [ ] Migration guide updated (if breaking changes)
- [ ] Examples are working and up to date

### Testing
- [ ] Manual testing completed for major features
- [ ] Performance benchmarks show no regression
- [ ] Integration tests pass
- [ ] Cross-platform testing completed (Linux, macOS, Windows)

### Version Management
- [ ] Version number follows semantic versioning
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `src/networkx_mcp/__version__.py`

## Release Process

1. **Ensure main branch is up to date**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Run the release script**
   ```bash
   ./scripts/release.sh
   ```

3. **Monitor GitHub Actions**
   - Check CI pipeline passes
   - Verify release workflow completes
   - Confirm package is published to PyPI

4. **Post-Release Tasks**
   - [ ] Verify release on GitHub releases page
   - [ ] Test installation from PyPI (`pip install networkx-mcp`)
   - [ ] Update documentation site
   - [ ] Announce release (if major version)
   - [ ] Close related issues and PRs

## Rollback Procedure

If issues are discovered after release:

1. **Delete the release tag**
   ```bash
   git tag -d v<version>
   git push origin :refs/tags/v<version>
   ```

2. **Yank from PyPI (if published)**
   ```bash
   pip install twine
   twine yank networkx-mcp <version>
   ```

3. **Fix issues and create new patch release**

## Release Types

### Patch Release (x.x.X)
- Bug fixes only
- No API changes
- No new features

### Minor Release (x.X.0)
- New features
- Backward compatible API changes
- Deprecations allowed

### Major Release (X.0.0)
- Breaking API changes
- Major refactoring
- Removal of deprecated features

## Communication

### Release Notes Template
```markdown
## NetworkX MCP Server v<version>

### What's New
- Feature 1
- Feature 2

### Improvements
- Improvement 1
- Improvement 2

### Bug Fixes
- Fix 1
- Fix 2

### Breaking Changes (if any)
- Change 1
- Migration: how to update

### Contributors
Thanks to all contributors!

### Full Changelog
See [CHANGELOG.md](CHANGELOG.md) for details.
```