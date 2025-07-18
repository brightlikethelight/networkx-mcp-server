# Repository Cleanup & Production-Grade Reorganization Plan

## Current Issues Analysis

### Root Directory Clutter (30+ files)

**CRITICAL**: Root has 30+ markdown files, making it unprofessional and hard to navigate

**Keep (Essential)**:

- README.md
- LICENSE
- CHANGELOG.md
- CONTRIBUTING.md
- pyproject.toml
- requirements.txt
- Dockerfile

**Archive to docs/archive/**:

- All historical documents (PHASE_*, TRANSFORMATION_*, MIGRATION_*, etc.)
- Development reports (MEMORY_BLOAT_*, PERFORMANCE_*, etc.)
- Launch documents (LAUNCH_*, SOCIAL_MEDIA_*, etc.)

**Remove (Obsolete)**:

- Multiple README variants (README_HONEST.md, README_MINIMAL.md)
- Multiple Dockerfile variants
- Development scripts and test files in root
- Build artifacts and temporary files

### Code Organization Issues

**Multiple Server Implementations** (Needs Consolidation):

- server_minimal.py (KEEP - main implementation)
- server.py (ARCHIVE - legacy full server)
- server_fastmcp.py (REMOVE - experimental)
- server_legacy.py (REMOVE - legacy)

**Overlapping Security Modules**:

- security/ (Legacy security)
- security_fortress/ (Current - KEEP)
- enterprise/ (Contains some security overlap)

**Test Organization** (Scattered):

- tests/ (Main location - KEEP)
- src/tests/ (Remove)
- Multiple test types mixed together

## Reorganization Plan

### 1. Root Directory Structure (Target: <15 files)

```
networkx-mcp-server/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── setup.cfg
├── .gitignore
├── src/
├── tests/
├── docs/
├── examples/
└── scripts/
```

### 2. Source Code Structure (Clean Modules)

```
src/networkx_mcp/
├── __init__.py
├── __version__.py
├── server_minimal.py (MAIN SERVER)
├── core/
│   ├── graph_operations.py
│   ├── algorithms.py
│   └── storage/
├── security_fortress/ (CURRENT SECURITY)
├── enterprise/
│   ├── auth.py
│   ├── monitoring.py
│   └── config.py
├── visualization/
├── io/
└── utils/
```

### 3. Documentation Structure

```
docs/
├── user-guide/
├── api/
├── deployment/
├── security/
├── enterprise/
├── examples/
└── archive/ (Historical documents)
```

### 4. Test Structure (Organized by Type)

```
tests/
├── unit/
├── integration/
├── security/
├── enterprise/
├── e2e/
└── fixtures/
```

## Implementation Steps

### Phase 1: Root Directory Cleanup

1. Create docs/archive/ directory
2. Move historical documents to archive
3. Remove obsolete files
4. Consolidate Docker files
5. Clean up build artifacts

### Phase 2: Code Consolidation

1. Remove legacy server implementations
2. Consolidate security modules
3. Clean up overlapping functionality
4. Improve import structure
5. Add proper **init**.py files

### Phase 3: Test Organization

1. Consolidate test directories
2. Organize by test type
3. Improve test coverage
4. Add integration tests
5. Clean up test dependencies

### Phase 4: Documentation Restructure

1. Archive historical docs
2. Update main README
3. Create deployment guides
4. Improve API documentation
5. Add enterprise guides

### Phase 5: Quality Improvements

1. Add comprehensive linting
2. Improve code coverage
3. Add performance tests
4. Security scanning integration
5. Automated quality gates
