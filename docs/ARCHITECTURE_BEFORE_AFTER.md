# Architecture: Before and After

## The Problem We Fixed

We discovered our "minimal" server was loading 900+ modules and using 118MB of RAM because `core/__init__.py` eagerly imported I/O handlers that most users never touched.

## BEFORE: Monolithic Bloat (v0.1.0-alpha.1)

```
┌─────────────────────────────────────────────────────────┐
│              MCP Server (118MB Total)                    │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │          Core Module (Always Loaded)            │     │
│  │                                                 │     │
│  │  server.py                                      │     │
│  │    └─→ core/__init__.py                        │     │
│  │         ├─→ graph_operations.py                │     │
│  │         ├─→ algorithms.py                      │     │
│  │         └─→ io/__init__.py  ⚠️ THE KILLER      │     │
│  │              └─→ io_handlers.py                │     │
│  │                   ├─→ import pandas    (+35MB) │     │
│  │                   ├─→ import scipy     (+15MB) │     │
│  │                   └─→ import numpy     (+17MB) │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  Result: EVERYONE pays 118MB even for basic graphs      │
└─────────────────────────────────────────────────────────┘

Import Chain of Death:
1. User: "I want to create a simple graph"
2. Server: "Sure! Let me load pandas, scipy, numpy, matplotlib..."
3. User: "But I just want add_edge()!"
4. Server: "Too bad, here's 900+ modules"
```

## AFTER: Modular Honesty (v0.1.0-alpha.2)

```
┌─────────────────────────────────────────────────────────┐
│            Minimal Server (54MB Default)                 │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │        Core Module (Always Loaded)              │     │
│  │                                                 │     │
│  │  server_minimal.py                              │     │
│  │    └─→ networkx (required)              (+20MB) │     │
│  │    └─→ asyncio, json (stdlib)           (+18MB) │     │
│  │                                                 │     │
│  │  core/__init__.py                               │     │
│  │    ├─→ graph_operations.py ✓                   │     │
│  │    ├─→ algorithms.py ✓                         │     │
│  │    └─→ get_io_handler() (lazy) 🔄              │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  ┌────────────────────────────────────────────────┐     │
│  │     Optional Extras (Install When Needed)       │     │
│  │                                                 │     │
│  │  [excel] → pandas, openpyxl            (+35MB) │     │
│  │  [scipy] → scipy, numpy                (+15MB) │     │
│  │  [viz]   → matplotlib                  (+8MB)  │     │
│  └────────────────────────────────────────────────┘     │
│                                                          │
│  Result: Pay only for what you use!                     │
└─────────────────────────────────────────────────────────┘

Lazy Loading Pattern:
1. User: "I want to create a simple graph"
2. Server: "Here's NetworkX, that's all you need (54MB)"
3. User: "Now I want to import from Excel"
4. Server: "Install [excel] extra first, then I'll load pandas"
```

## Memory Breakdown Comparison

### Before (Monolithic)

```
Python interpreter          16MB ████
NetworkX                    24MB ██████
NumPy                       17MB ████
Pandas                      35MB █████████
SciPy                       15MB ████
Other imports               11MB ███
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                      118MB ███████████████████████████████
```

### After (Modular)

```
MINIMAL (Default):
Python interpreter          16MB ████
NetworkX                    20MB █████
Server + asyncio            18MB ████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                       54MB █████████████

WITH EXCEL:
Minimal                     54MB █████████████
+ Pandas                    35MB █████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                       89MB ██████████████████████

FULL:
Minimal                     54MB █████████████
+ Pandas                    35MB █████████
+ SciPy                     15MB ████
+ Matplotlib                14MB ███
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL                      118MB ███████████████████████████████
```

## Code Changes

### The Fatal Import (Before)

```python
# core/__init__.py (v0.1.0-alpha.1)
from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.core.io import GraphIOHandler  # ← LOADS PANDAS IMMEDIATELY!

__all__ = ["GraphAlgorithms", "GraphIOHandler", "GraphManager"]
```

### The Fix (After)

```python
# core/__init__.py (v0.1.0-alpha.2)
from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager

# DO NOT import GraphIOHandler here - it loads pandas (+35MB)!
__all__ = ["GraphAlgorithms", "GraphManager", "get_io_handler"]

def get_io_handler():
    """Lazy load IO handler only when needed."""
    from networkx_mcp.core.io import GraphIOHandler
    return GraphIOHandler
```

## Installation Flow

### Before

```
pip install networkx-mcp
│
└─→ Installs EVERYTHING
    ├─→ networkx (required) ✓
    ├─→ pandas (forced) ✗
    ├─→ scipy (forced) ✗
    └─→ numpy (forced) ✗

Result: 118MB for everyone
```

### After

```
pip install networkx-mcp
│
└─→ Installs MINIMAL
    └─→ networkx (required) ✓

Result: 54MB default

pip install networkx-mcp[excel]
│
└─→ Installs MINIMAL + EXCEL
    ├─→ networkx (required) ✓
    └─→ pandas (optional) ✓

Result: 89MB when needed

pip install networkx-mcp[full]
│
└─→ Installs EVERYTHING
    ├─→ networkx (required) ✓
    ├─→ pandas (optional) ✓
    ├─→ scipy (optional) ✓
    └─→ matplotlib (optional) ✓

Result: 118MB by choice
```

## Key Architectural Principles

1. **Minimal means minimal**: Don't force heavyweight dependencies
2. **Pay for what you use**: Optional features should be optional
3. **Lazy loading**: Import expensive modules only when needed
4. **Honest defaults**: Most users don't need Excel import
5. **Clear boundaries**: Separate core from extras

## Lessons Learned

- One innocent import can cascade into 900+ modules
- "Just in case" dependencies are architectural debt
- Measure memory usage, don't assume
- Be honest about what "minimal" means
- Users appreciate choice

The architecture is now honest, modular, and actually minimal.
