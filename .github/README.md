# NetworkX MCP Server

A comprehensive Model Context Protocol (MCP) server providing advanced graph analysis capabilities using NetworkX.

## 🚀 Features

- **Complete MCP Implementation**: Full Model Context Protocol support with Tools, Resources, and Prompts
- **Modular Architecture**: Clean, maintainable codebase with 35+ focused modules
- **Advanced Graph Analysis**: Comprehensive suite of graph algorithms and analytics
- **Production Ready**: Enterprise-grade security, monitoring, and scalability features
- **Developer Friendly**: Extensive documentation, testing, and development tools

## 🏗️ Architecture

The server follows a clean modular architecture:

```
├── Core Layer          # Basic graph operations and MCP server
├── Handler Layer       # Function organization and re-exports
├── Advanced Layer      # Specialized algorithms and features
└── Supporting Layer    # Monitoring, security, and infrastructure
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architectural documentation.

## 📦 Quick Start

### Installation

```bash
git clone https://github.com/username/networkx-mcp-server.git
cd networkx-mcp-server
pip install -e .
```

### Basic Usage

```python
from networkx_mcp.server import create_graph, add_nodes, add_edges

# Create a graph
result = create_graph("my_graph", "undirected")

# Add nodes and edges
add_nodes("my_graph", ["A", "B", "C"])
add_edges("my_graph", [("A", "B"), ("B", "C")])
```

### Running the Server

```bash
# Start the MCP server
python -m networkx_mcp

# Or use the development script
./run_tests.sh
```

## 🧪 Testing

The project maintains 80%+ test coverage with comprehensive test suites:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/networkx_mcp --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests
```

## 📖 Documentation

- [Architecture Overview](ARCHITECTURE.md) - Complete system architecture
- [Module Structure](docs/MODULE_STRUCTURE.md) - Detailed module organization
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Developer handbook
- [API Documentation](docs/api/) - Detailed API reference

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/DEVELOPMENT_GUIDE.md) for:

- Setting up the development environment
- Code standards and conventions
- Testing requirements
- Submission guidelines

### Quick Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run the test suite
pytest
```

## 🏆 Quality Standards

This project maintains high quality standards:

- **Code Quality**: Automated formatting with ruff, black, and isort
- **Type Safety**: Comprehensive type hints with mypy validation
- **Security**: Bandit security scanning and vulnerability checks
- **Testing**: 80%+ test coverage with multiple test categories
- **Documentation**: Comprehensive documentation and examples

## 📋 Requirements

- Python 3.11+
- NetworkX 3.0+
- FastMCP (or compatible MCP implementation)

See [pyproject.toml](pyproject.toml) for complete dependency list.

## 🚀 Deployment

### Docker

```bash
# Build and run with Docker
docker build -t networkx-mcp-server .
docker run -p 8000:8000 networkx-mcp-server
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/
```

See [deployment documentation](docs/deployment/) for production deployment guides.

## 📊 Performance

The server is optimized for performance:

- **Modular Design**: Efficient memory usage and fast load times
- **Algorithm Optimization**: Optimized implementations for large graphs
- **Monitoring**: Built-in performance metrics and health checks
- **Scalability**: Stateless design supporting horizontal scaling

## 🔒 Security

Security is a top priority:

- **Input Validation**: Comprehensive input sanitization and validation
- **Access Control**: Authentication and authorization layers
- **Audit Logging**: Complete audit trail for security events
- **Vulnerability Scanning**: Automated dependency vulnerability checks

## 📈 Monitoring

Built-in observability features:

- **Health Checks**: Comprehensive health monitoring endpoints
- **Metrics**: Performance and usage metrics collection
- **Tracing**: Distributed tracing support
- **Logging**: Structured logging with configurable levels

## 🗂️ Project Structure

```
networkx-mcp-server/
├── src/networkx_mcp/       # Main source code
│   ├── core/               # Core graph operations
│   ├── handlers/           # Function handlers
│   ├── advanced/           # Advanced algorithms
│   ├── monitoring/         # Monitoring and observability
│   └── security/           # Security features
├── tests/                  # Comprehensive test suite
├── docs/                   # Documentation
├── scripts/                # Development and deployment scripts
└── examples/               # Usage examples
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NetworkX team for the excellent graph analysis library
- FastMCP team for the Model Context Protocol implementation
- Contributors and users of this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/username/networkx-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/username/networkx-mcp-server/discussions)
- **Documentation**: [Project Documentation](docs/)

---

**Built with ❤️ for the graph analysis community**
