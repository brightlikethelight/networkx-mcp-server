#!/bin/bash
echo "🧪 Running NetworkX MCP Server Tests..."
echo "======================================"

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Unit tests with coverage
echo "📊 Running unit tests with coverage..."
pytest tests/unit/ -v --cov=src/networkx_mcp --cov-report=html --cov-report=term --timeout=30

# Integration tests
echo "🔄 Running integration tests..."
pytest tests/integration/ -v --timeout=60

# Core operations tests
echo "🔧 Running core operations tests..."
pytest tests/test_core_operations.py -v

# Feature audit
echo "🔍 Running feature audit..."
python tests/test_feature_audit.py

# Generate final report
echo "======================================"
echo "✅ Test run complete!"
echo "📊 Coverage report: htmlcov/index.html"
echo "🔍 Feature audit results above"
