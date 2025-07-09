# Migration Guide: v0.1.0-alpha.1 → v0.1.0-alpha.2

## Overview

Version 0.1.0-alpha.2 fixes a critical architectural flaw where the "minimal" server loaded 118MB of scientific Python libraries that most users never needed. This is a breaking change that makes the server actually minimal.

## Breaking Changes

### 1. I/O Handlers No Longer Auto-Imported

The biggest breaking change is that `GraphIOHandler` is no longer automatically available.

**Before (v0.1.0-alpha.1)**:
```python
from networkx_mcp.core import GraphIOHandler
handler = GraphIOHandler()  # Always worked, but loaded pandas (+35MB)
```

**After (v0.1.0-alpha.2)**:
```python
# Option 1: Lazy import (recommended)
from networkx_mcp.core import get_io_handler
GraphIOHandler = get_io_handler()  # Only loads pandas when called
handler = GraphIOHandler()

# Option 2: Direct import (requires pandas installed)
from networkx_mcp.core.io_handlers import GraphIOHandler
handler = GraphIOHandler()  # ImportError if pandas not installed
```

### 2. Default Installation No Longer Includes Pandas

**Before**:
```bash
pip install networkx-mcp  # Installed everything (118MB)
```

**After**:
```bash
pip install networkx-mcp          # Minimal only (54MB)
pip install networkx-mcp[excel]   # If you need Excel/CSV (89MB)
pip install networkx-mcp[full]    # Everything like before (118MB)
```

### 3. Excel/CSV Operations May Fail

If your code uses Excel or CSV import/export, it will now fail unless you install the extras.

**Error you'll see**:
```
ImportError: Excel import requires pandas. Install with: pip install networkx-mcp[excel]
```

**Fix**:
```bash
pip install networkx-mcp[excel]
```

## Memory Impact

Your application's memory usage will change:

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Basic graph operations | 118MB | 54MB | 64MB (54%) |
| With Excel/CSV | 118MB | 89MB | 29MB (25%) |
| Full features | 118MB | 118MB | 0MB |

## Migration Steps

### Step 1: Check Your Imports

Search your codebase for:
```bash
grep -r "GraphIOHandler" .
grep -r "import.*pandas" .
grep -r "import.*scipy" .
```

### Step 2: Update Import Statements

If you find `GraphIOHandler` imports:
```python
# Old
from networkx_mcp.core import GraphIOHandler

# New (if you always need it)
from networkx_mcp.core import get_io_handler
GraphIOHandler = get_io_handler()

# New (if rarely needed)
def load_from_excel(filepath):
    from networkx_mcp.core import get_io_handler
    GraphIOHandler = get_io_handler()
    return GraphIOHandler.import_from_file(filepath, 'excel')
```

### Step 3: Update Installation

Check what features you actually use:

- **Just creating graphs and running algorithms?** → Use minimal install
- **Need Excel/CSV import?** → Add `[excel]`
- **Need sparse matrices?** → Add `[scipy]` 
- **Need everything?** → Use `[full]`

### Step 4: Test Your Application

1. Install the minimal version first
2. Run your test suite
3. If you get ImportError for pandas/scipy, add the needed extras
4. Only install what fails

## Common Scenarios

### Scenario 1: "I just use basic graph operations"

No changes needed! Your app just got 64MB lighter.

### Scenario 2: "I occasionally import from Excel"

```python
# Wrap imports in try/except
try:
    from networkx_mcp.core import get_io_handler
    GraphIOHandler = get_io_handler()
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False
    
def load_excel(filepath):
    if not HAS_EXCEL:
        raise ImportError("Excel support not installed. Run: pip install networkx-mcp[excel]")
    return GraphIOHandler.import_from_file(filepath, 'excel')
```

### Scenario 3: "I need everything"

```bash
pip install networkx-mcp[full]
```
No code changes needed, but you're still using 118MB.

## Benefits of Migrating

1. **54% memory reduction** for most users
2. **Faster startup** (no pandas import)
3. **Cleaner dependencies** (only what you need)
4. **Honest architecture** (actually minimal)

## Rollback Plan

If you need to quickly rollback:

```bash
pip install networkx-mcp[full]
```

This installs all dependencies like v0.1.0-alpha.1, but you should plan to migrate properly.

## Getting Help

- Check if you actually need pandas/scipy before installing them
- Most graph operations don't need Excel import
- Consider using JSON format instead of Excel (no pandas needed)
- Report issues with migration to our issue tracker

## Future Considerations

We may further modularize in future versions. Start thinking about what features you actually use vs what you have installed "just in case".