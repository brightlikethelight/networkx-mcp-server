# Pull Request

## 📋 Summary

<!-- Provide a clear and concise summary of your changes -->

**Type of Change:**

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧪 Test coverage improvement
- [ ] 🔒 Security enhancement
- [ ] 🚀 CI/CD improvement

## 🎯 Related Issues

<!-- Link to related issues -->
Fixes #(issue number)
Relates to #(issue number)

## 🚀 Changes Made

<!-- Describe the changes made in detail -->

### Added

- New feature X that allows users to...
- New MCP tool `tool_name` for...

### Changed

- Modified algorithm Y to improve performance by...
- Updated API endpoint Z to...

### Fixed

- Resolved issue where...
- Fixed memory leak in...

### Removed

- Deprecated function A because...
- Removed unused dependency B...

## 🧪 Testing

<!-- Describe how you tested your changes -->

### Test Coverage

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Property-based tests added/updated
- [ ] Security tests added/updated
- [ ] Performance tests added/updated

### Manual Testing

- [ ] Tested on local development environment
- [ ] Tested with different MCP clients
- [ ] Tested with various graph sizes
- [ ] Verified backward compatibility

### Test Results

```bash
# Include relevant test results
pytest tests/ --cov=src/networkx_mcp --cov-report=term-missing
# Coverage: 95.8%
# All tests passed
```

## 📊 Performance Impact

<!-- If applicable, describe performance implications -->

### Benchmarks

- [ ] No performance impact
- [ ] Performance improvement: X% faster
- [ ] Minor performance impact: X% slower (justified because...)
- [ ] Benchmarks added/updated

### Memory Usage

- [ ] No memory impact
- [ ] Reduced memory usage
- [ ] Increased memory usage (justified because...)

## 🔄 Breaking Changes

<!-- List any breaking changes and migration instructions -->

**Breaking Changes:**

- None ✅
- Changed API endpoint X (migration: ...)
- Modified function signature Y (migration: ...)

**Migration Guide:**

```python
# Before
old_function(param1, param2)

# After
new_function(param1, param2, new_param3)
```

## 📚 Documentation

<!-- Documentation updates -->

- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] README updated
- [ ] Example code provided
- [ ] Migration guide created (if breaking changes)
- [ ] Changelog updated

## ✅ Pre-submission Checklist

<!-- Ensure all items are checked before submitting -->

### Code Quality

- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No TODO comments left in production code
- [ ] Error handling implemented appropriately
- [ ] Logging added where appropriate

### Testing & Validation

- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Test coverage maintained/improved
- [ ] Manual testing completed
- [ ] No regressions introduced

### Dependencies & Security

- [ ] No new dependencies added (or justified in description)
- [ ] Security implications considered
- [ ] No sensitive data exposed
- [ ] Input validation implemented where needed

### Git & Process

- [ ] Commits follow [Conventional Commits](https://conventionalcommits.org/) format
- [ ] Branch is up to date with main
- [ ] Clean commit history (squashed if needed)
- [ ] PR title clearly describes the change

## 🔒 Security Considerations

<!-- Address any security implications -->

- [ ] No security implications
- [ ] Security review completed
- [ ] Potential security issues addressed:
  - Input validation for...
  - Authentication/authorization for...
  - Data sanitization for...

## 🌍 Deployment Considerations

<!-- Consider deployment implications -->

- [ ] No deployment changes needed
- [ ] Environment variables added/changed
- [ ] Database migrations required
- [ ] Configuration updates needed
- [ ] Service restart required

## 📋 Review Checklist for Maintainers

<!-- For maintainer use -->

- [ ] Code quality meets standards
- [ ] Tests are comprehensive
- [ ] Documentation is adequate
- [ ] Performance impact acceptable
- [ ] Security review completed
- [ ] Breaking changes properly handled
- [ ] CI/CD passes

## 🤝 Collaboration

<!-- Thank contributors and reviewers -->

**Contributors:**

- @username (implementation)
- @username (testing)
- @username (review)

**Reviewers Requested:**

- @maintainer1
- @domain-expert

## 📝 Additional Notes

<!-- Any additional context or notes -->

### Implementation Details

- Chose approach X over Y because...
- Algorithm complexity is O(n) which is acceptable because...
- Used library Z for...

### Future Work

- Follow-up issue needed for...
- Could be extended to support...
- Performance could be further improved by...

### Questions for Reviewers

- Should we consider alternative approach X?
- Is the performance impact acceptable?
- Any concerns about the API design?

---

**Ready for Review:** <!-- Check when ready -->

- [ ] This PR is ready for review
- [ ] This is a draft PR (work in progress)

**Post-merge Tasks:** <!-- If applicable -->

- [ ] Update documentation website
- [ ] Announce feature in discussions
- [ ] Create follow-up issues
- [ ] Update examples/tutorials
