# Git Version Control Strategy for NetworkX MCP Server

## Version Control Principles

### 1. Commit Standards
All commits MUST follow the Conventional Commits specification v1.0.0:

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: New feature implementation
- **fix**: Bug fixes
- **docs**: Documentation changes
- **style**: Code style/formatting (no logic changes)
- **refactor**: Code restructuring without behavior change
- **perf**: Performance improvements
- **test**: Test additions or modifications
- **build**: Build system or dependency changes
- **ci**: CI/CD configuration changes
- **chore**: Maintenance tasks

### Scopes
- **monitoring**: Monitoring and observability features
- **ci/cd**: CI/CD pipeline and workflows
- **core**: Core graph operations
- **mcp**: MCP protocol implementation
- **tools**: MCP tools and utilities
- **deps**: Dependencies and requirements
- **security**: Security-related changes

## Branch Strategy

### Main Branches
- **main**: Production-ready code
- **develop**: Integration branch for features
- **release/**: Release preparation branches

### Supporting Branches
- **feature/**: New features (`feature/add-graph-visualization`)
- **fix/**: Bug fixes (`fix/resolve-memory-leak`)
- **hotfix/**: Critical production fixes
- **docs/**: Documentation updates
- **experiment/**: Experimental features

## Version Control Workflow

### 1. Pre-Commit Checks
```bash
# Before every commit, ensure:
1. All tests pass: pytest tests/
2. Code is formatted: ruff format .
3. No linting errors: ruff check .
4. Type checks pass: mypy .
5. Security scan clean: bandit -r src/
```

### 2. Commit Process
```bash
# Stage changes selectively
git add -p

# Write descriptive commit
git commit -m "type(scope): description

- Detailed explanation of what changed
- Why this change was necessary
- Any breaking changes or migrations required

Closes #issue-number"
```

### 3. Pull Request Standards
- **Title**: Follow commit message format
- **Description**: Include:
  - Problem being solved
  - Solution approach
  - Testing performed
  - Screenshots/logs if applicable
  - Breaking changes
  - Migration guide

### 4. Code Review Requirements
- Minimum 1 approval required
- All CI checks must pass
- No merge conflicts
- Coverage must not decrease
- Security scan must pass

## Versioning Strategy

### Semantic Versioning (SemVer)
Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Version Bumping Rules
```python
# Automated version bumping based on commits
if "BREAKING CHANGE" in commit_message:
    bump_major()
elif commit_type == "feat":
    bump_minor()
elif commit_type == "fix":
    bump_patch()
```

## Git Hooks

### Pre-commit Hook
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run tests
pytest tests/working/ --quiet

# Check formatting
ruff format --check .

# Security scan
bandit -r src/ --severity-level medium

if [ $? -ne 0 ]; then
  echo "Pre-commit checks failed!"
  exit 1
fi
```

### Commit-msg Hook
```bash
#!/bin/sh
# .git/hooks/commit-msg

# Validate commit message format
commit_regex='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)(\([a-z/-]+\))?: .{1,50}'

if ! grep -qE "$commit_regex" "$1"; then
    echo "Invalid commit message format!"
    echo "Format: type(scope): subject"
    exit 1
fi
```

### Post-commit Hook
```bash
#!/bin/sh
# .git/hooks/post-commit

# Log commit to monitoring
python -c "
from networkx_mcp.monitoring.dora_metrics import dora_collector
import subprocess

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
author = subprocess.check_output(['git', 'log', '-1', '--format=%ae']).decode().strip()

# Track commit for DORA metrics
dora_collector.track_commit(commit, author)
"
```

## Release Process

### 1. Release Preparation
```bash
# Create release branch
git checkout -b release/v1.2.0

# Update version
bump2version minor

# Update CHANGELOG
git add CHANGELOG.md
git commit -m "chore(release): prepare v1.2.0"
```

### 2. Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version bumped
- [ ] Security scan clean
- [ ] Performance benchmarks run
- [ ] Migration guide prepared (if needed)

### 3. Release Tagging
```bash
# Tag release
git tag -a v1.2.0 -m "Release v1.2.0

Features:
- Feature 1
- Feature 2

Fixes:
- Fix 1
- Fix 2

See CHANGELOG.md for details"

# Push tag
git push origin v1.2.0
```

## History Management

### Commit Squashing
```bash
# Interactive rebase for clean history
git rebase -i HEAD~3

# Squash related commits
pick abc1234 feat: add feature
squash def5678 fix: typo in feature
squash ghi9012 docs: update feature docs
```

### History Rewriting Rules
- **NEVER** rewrite public history (main branch)
- **ALWAYS** squash WIP commits before merging
- **KEEP** meaningful commit messages

## Backup and Recovery

### Regular Backups
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
git bundle create backups/repo-$DATE.bundle --all
aws s3 cp backups/repo-$DATE.bundle s3://backups/
```

### Recovery Process
```bash
# Restore from bundle
git clone backups/repo-20240826.bundle restored-repo
cd restored-repo
git remote set-url origin https://github.com/user/repo.git
```

## CI/CD Integration

### Automated Version Control Checks
```yaml
# .github/workflows/version-control.yml
name: Version Control Enforcement

on: [push, pull_request]

jobs:
  enforce-standards:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check commit messages
        uses: wagoid/commitlint-github-action@v5
        
      - name: Check branch naming
        run: |
          branch=$(echo ${GITHUB_REF#refs/heads/})
          if ! [[ "$branch" =~ ^(main|develop|feature/.*|fix/.*|hotfix/.*|docs/.*|release/.*|experiment/.*)$ ]]; then
            echo "Invalid branch name: $branch"
            exit 1
          fi
          
      - name: Generate commit report
        run: |
          git log --format="%h %s" --since="1 week ago" > commit-report.txt
          python scripts/analyze_commits.py commit-report.txt
```

## Best Practices

### DO's ✅
- Write atomic commits (one logical change per commit)
- Use descriptive branch names
- Sign commits with GPG when possible
- Keep commit messages under 72 characters
- Reference issues in commits
- Review your own PR before requesting review
- Keep PRs focused and small (<500 lines)
- Update documentation with code changes

### DON'Ts ❌
- Don't commit sensitive data (keys, passwords)
- Don't commit generated files
- Don't force push to main
- Don't merge without review
- Don't commit broken code
- Don't mix refactoring with features
- Don't commit large binary files
- Don't ignore CI failures

## Monitoring and Metrics

### Track Version Control Health
```python
# Monitor commit quality
metrics = {
    "commit_frequency": calculate_daily_commits(),
    "pr_size_average": calculate_average_pr_size(),
    "review_turnaround": calculate_review_time(),
    "commit_message_quality": analyze_commit_messages(),
    "branch_lifetime": calculate_branch_age(),
}

if metrics["pr_size_average"] > 500:
    alert("PRs too large - consider splitting")
    
if metrics["branch_lifetime"] > 7:
    alert("Long-lived branches detected")
```

## Emergency Procedures

### Rollback Process
```bash
# Quick rollback to previous version
git revert HEAD
git push origin main

# Or reset to specific commit
git reset --hard <commit-hash>
git push --force-with-lease origin main
```

### Conflict Resolution
```bash
# Standard conflict resolution
git checkout develop
git pull origin develop
git checkout feature/branch
git rebase develop
# Resolve conflicts
git add .
git rebase --continue
```

## Training and Onboarding

### New Developer Checklist
- [ ] Read this version control strategy
- [ ] Set up git hooks locally
- [ ] Configure git aliases
- [ ] Practice commit message format
- [ ] Review recent PR examples
- [ ] Complete git training module

### Git Aliases for Productivity
```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'
git config --global alias.amend 'commit --amend'
```

---

*Last Updated: December 2024*
*Version: 1.0.0*
*Maintained by: NetworkX MCP Server Team*