# Release Checklist for v2.0.0

## Pre-release Tasks

### Code Quality ✓
- [x] All tests pass
- [x] Type hints complete
- [x] Documentation updated
- [x] CHANGELOG.md updated
- [x] Version bumped in pyproject.toml

### Git Repository ✓
- [x] All changes committed
- [x] Git history cleaned (Claude references removed)
- [x] Branch is up to date with main

### Documentation
- [ ] README.md updated with new features
- [ ] MIGRATION_NOTES.md finalized
- [ ] API documentation generated
- [ ] Examples updated

### Testing
- [ ] Manual testing of server_v2.py
- [ ] Backward compatibility verified
- [ ] Performance benchmarks run
- [ ] Security scan completed

## Release Process

### 1. Build Package
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Check the distribution
twine check dist/*
```

### 2. Test Installation
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/networkx_mcp_server-2.0.0-py3-none-any.whl

# Test imports
python -c "from networkx_mcp.server_v2 import NetworkXMCPServer; print('✓ Import successful')"

# Deactivate
deactivate
```

### 3. Create GitHub Release
- Tag: v2.0.0
- Title: NetworkX MCP Server 2.0.0 - Complete MCP Implementation
- Description: Include highlights from CHANGELOG.md
- Attach wheel and source distribution

### 4. Publish to PyPI
```bash
# Test PyPI first
twine upload --repository testpypi dist/*

# Production PyPI
twine upload dist/*
```

### 5. Post-release
- [ ] Verify PyPI page
- [ ] Test pip install from PyPI
- [ ] Update documentation site
- [ ] Announce on relevant channels

## Rollback Plan

If issues are found:
1. Yank package from PyPI: `pip install -U twine && twine yank networkx-mcp-server==2.0.0`
2. Fix issues in new branch
3. Release as 2.0.1 with fixes
4. Update documentation

## Notes

- PyPI credentials should be configured in ~/.pypirc
- Use API tokens for authentication
- Test thoroughly before production release
- Keep backup of v1.0.0 for rollback
