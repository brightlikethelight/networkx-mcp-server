# Git Commit Conventions

This repository follows **Conventional Commits** specification for clear and consistent commit history.

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type (Required)

Must be one of the following:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, missing semi-colons, etc)
- **refactor**: Code refactoring without changing functionality
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Changes affecting build system or dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't affect src or test files
- **revert**: Reverts a previous commit

### Scope (Optional)

The scope should be the module or component name:

- **server**: Core server functionality
- **monitoring**: CI/CD monitoring and health checks
- **academic**: Academic features (DOI, citations)
- **security**: Security-related changes
- **docker**: Docker configuration
- **deps**: Dependencies
- **api**: API changes
- **cli**: Command-line interface
- **config**: Configuration changes

### Subject (Required)

- Use imperative mood ("add" not "adds" or "added")
- Don't capitalize the first letter
- No period at the end
- Maximum 50 characters

### Body (Optional)

- Use imperative mood
- Explain *what* and *why* vs *how*
- Wrap at 72 characters
- Separate from subject with blank line

### Footer (Optional)

- Reference GitHub issues: `Fixes #123`, `Closes #456`
- Note breaking changes: `BREAKING CHANGE: <description>`
- Co-authors: `Co-authored-by: Name <email>`

## Examples

### Simple Feature

```
feat(monitoring): add Slack webhook integration
```

### Bug Fix with Issue

```
fix(server): handle empty graph edge cases

The server now properly validates graph operations when
the graph is empty or contains no edges.

Fixes #789
```

### Breaking Change

```
refactor(api)!: rename graph_name to graph_id

BREAKING CHANGE: All API endpoints now use graph_id
instead of graph_name for consistency with MCP spec.

Migration guide available in docs/MIGRATION.md
```

### Multiple Scopes

```
feat(monitoring,ci): enhance pipeline health tracking

- Add test coverage trends
- Track performance benchmarks
- Monitor dependency vulnerabilities
```

### Revert

```
revert: feat(monitoring): add Slack webhook integration

This reverts commit 123abc4.
Reason: Webhook causing rate limit issues
```

## Commit Signing

For security, all commits should be signed:

```bash
# Configure GPG signing
git config --local commit.gpgsign true
git config --local user.signingkey <your-key-id>
```

## Pre-commit Hooks

Ensure commits pass all checks:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually before committing
pre-commit run --all-files
```

## Branch Naming

Use descriptive branch names:

- `feat/<feature-name>` - New features
- `fix/<issue-number>-<description>` - Bug fixes
- `docs/<what-docs>` - Documentation
- `refactor/<what-refactor>` - Code refactoring
- `test/<what-test>` - Test additions/changes
- `ci/<what-ci>` - CI/CD changes

## Pull Request Guidelines

1. **Title**: Use same format as commits
2. **Description**: Include:
   - What changes were made
   - Why they were necessary
   - How to test the changes
   - Screenshots (if UI changes)
3. **Link Issues**: Reference related issues
4. **Tests**: Ensure all tests pass
5. **Documentation**: Update if needed

## Commit Frequency

- **Atomic commits**: One logical change per commit
- **Frequent commits**: Commit at least daily when actively developing
- **WIP commits**: Use `wip:` prefix for work-in-progress (squash before merge)

## Tools

### Commitizen

```bash
# Interactive commit helper
npm install -g commitizen
git cz
```

### Commitlint

```bash
# Validate commit messages
npm install -g @commitlint/cli @commitlint/config-conventional
echo "module.exports = {extends: ['@commitlint/config-conventional']}" > commitlint.config.js
```

### Git Aliases

```bash
# Useful aliases
git config --global alias.cm "commit -m"
git config --global alias.ca "commit --amend"
git config --global alias.unstage "reset HEAD --"
git config --global alias.last "log -1 HEAD"
git config --global alias.visual "log --oneline --graph --decorate"
```

## Resources

- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Angular Commit Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
