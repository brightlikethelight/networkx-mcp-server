#!/usr/bin/env python3
"""Development environment setup script for NetworkX MCP Server.

This script automates the setup of a complete development environment including:
- Virtual environment creation
- Development dependencies installation
- Pre-commit hooks setup
- IDE configuration
- Development database/services
- Testing environment validation
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


# ANSI color codes for better output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_step(message: str):
    """Print a step message with formatting."""
    print(f"{Colors.BLUE}🔧 {message}{Colors.END}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def run_command(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=cwd, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(e.stderr)
        if check:
            raise
        return e


class DevEnvironmentSetup:
    """Development environment setup coordinator."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.python_executable = sys.executable
        self.is_windows = platform.system() == "Windows"
        self.venv_path = project_root / "venv"

    def check_prerequisites(self):
        """Check that required tools are available."""
        print_step("Checking prerequisites")

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 11):
            print_error(
                f"Python 3.11+ required, found {python_version.major}.{python_version.minor}"
            )
            sys.exit(1)
        print_success(
            f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓"
        )

        # Check Git
        try:
            result = run_command(["git", "--version"])
            print_success("Git available ✓")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print_error("Git not found. Please install Git.")
            sys.exit(1)

        # Check for recommended tools
        recommended_tools = {
            "docker": "Docker for containerized services",
            "redis-server": "Redis for caching (optional)",
            "node": "Node.js for documentation (optional)",
        }

        for tool, description in recommended_tools.items():
            try:
                run_command([tool, "--version"], check=False)
                print_success(f"{tool} available ✓")
            except FileNotFoundError:
                print_warning(f"{tool} not found - {description}")

    def create_virtual_environment(self):
        """Create and activate virtual environment."""
        print_step("Setting up virtual environment")

        if self.venv_path.exists():
            print_warning("Virtual environment already exists")
            return

        # Create virtual environment
        run_command([self.python_executable, "-m", "venv", str(self.venv_path)])
        print_success("Virtual environment created")

        # Get activation script path
        if self.is_windows:
            activate_script = self.venv_path / "Scripts" / "activate.bat"
            pip_executable = self.venv_path / "Scripts" / "pip.exe"
        else:
            activate_script = self.venv_path / "bin" / "activate"
            pip_executable = self.venv_path / "bin" / "pip"

        print_success(f"Activate with: {activate_script}")

        # Upgrade pip
        run_command([str(pip_executable), "install", "--upgrade", "pip"])
        print_success("pip upgraded")

    def install_dependencies(self):
        """Install development dependencies."""
        print_step("Installing dependencies")

        # Get pip executable
        if self.is_windows:
            pip_executable = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_executable = self.venv_path / "bin" / "pip"

        # Install development dependencies
        run_command([str(pip_executable), "install", "-e", ".[dev,all]"])
        print_success("Development dependencies installed")

        # Install additional development tools
        dev_tools = [
            "ipython",  # Better REPL
            "ipdb",  # Better debugger
            "memory-profiler",  # Memory profiling
            "py-spy",  # CPU profiling
            "jupyterlab",  # Notebooks
            "commitizen",  # Conventional commits
        ]

        for tool in dev_tools:
            try:
                run_command([str(pip_executable), "install", tool])
                print_success(f"Installed {tool}")
            except subprocess.CalledProcessError:
                print_warning(f"Failed to install {tool}")

    def setup_pre_commit_hooks(self):
        """Set up pre-commit hooks."""
        print_step("Setting up pre-commit hooks")

        if self.is_windows:
            python_executable = self.venv_path / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_path / "bin" / "python"

        try:
            # Install pre-commit hooks
            run_command([str(python_executable), "-m", "pre_commit", "install"])
            run_command(
                [
                    str(python_executable),
                    "-m",
                    "pre_commit",
                    "install",
                    "--hook-type",
                    "pre-push",
                ]
            )
            print_success("Pre-commit hooks installed")

            # Run hooks on all files to verify setup
            print_step("Running pre-commit on all files")
            run_command(
                [str(python_executable), "-m", "pre_commit", "run", "--all-files"],
                check=False,
            )

        except subprocess.CalledProcessError:
            print_warning("Pre-commit setup failed - continuing without hooks")

    def create_ide_config(self):
        """Create IDE configuration files."""
        print_step("Creating IDE configuration")

        # VSCode configuration
        vscode_dir = self.project_root / ".vscode"
        vscode_dir.mkdir(exist_ok=True)

        # VSCode settings
        settings = {
            "python.defaultInterpreterPath": str(self.venv_path / "bin" / "python"),
            "python.terminal.activateEnvironment": True,
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": False,
            "python.linting.ruffEnabled": True,
            "python.linting.mypyEnabled": True,
            "python.formatting.provider": "black",
            "python.testing.pytestEnabled": True,
            "python.testing.unittestEnabled": False,
            "files.exclude": {
                "**/__pycache__": True,
                "**/.pytest_cache": True,
                "**/.mypy_cache": True,
                "**/.ruff_cache": True,
                "**/venv": True,
                "**/.venv": True,
                "**/node_modules": True,
            },
            "editor.formatOnSave": True,
            "editor.codeActionsOnSave": {"source.organizeImports": True},
        }

        with open(vscode_dir / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)

        # VSCode extensions recommendations
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.black-formatter",
                "charliermarsh.ruff",
                "ms-python.mypy-type-checker",
                "ms-python.pytest",
                "redhat.vscode-yaml",
                "yzhang.markdown-all-in-one",
                "ms-vscode.vscode-json",
                "GitHub.copilot",
                "ms-python.debugpy",
            ]
        }

        with open(vscode_dir / "extensions.json", "w") as f:
            json.dump(extensions, f, indent=2)

        # Launch configuration for debugging
        launch_config = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "Run MCP Server",
                    "type": "python",
                    "request": "launch",
                    "module": "networkx_mcp.server",
                    "console": "integratedTerminal",
                    "env": {"MCP_LOG_LEVEL": "DEBUG"},
                },
                {
                    "name": "Run Tests",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["tests/unit/", "-v"],
                    "console": "integratedTerminal",
                },
                {
                    "name": "Debug Test",
                    "type": "python",
                    "request": "launch",
                    "module": "pytest",
                    "args": ["${file}", "-v", "-s"],
                    "console": "integratedTerminal",
                },
            ],
        }

        with open(vscode_dir / "launch.json", "w") as f:
            json.dump(launch_config, f, indent=2)

        print_success("VSCode configuration created")

        # PyCharm configuration hint
        pycharm_hint = """
# PyCharm Configuration
For PyCharm users:
1. File -> Settings -> Project -> Python Interpreter
2. Add interpreter -> Existing environment
3. Select: {venv_python}
4. Enable pytest as test runner
5. Configure code style to use Black formatter
        """.format(
            venv_python=self.venv_path / "bin" / "python"
        )

        with open(self.project_root / "PYCHARM_SETUP.md", "w") as f:
            f.write(pycharm_hint)

    def setup_development_services(self):
        """Set up development services like Redis."""
        print_step("Setting up development services")

        # Create docker-compose for development
        docker_compose = """
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  redis-commander:
    image: rediscommander/redis-commander:latest
    environment:
      - REDIS_HOSTS=local:redis:6379
    ports:
      - "8081:8081"
    depends_on:
      - redis
      
volumes:
  redis_data:
"""

        with open(self.project_root / "docker-compose.dev.yml", "w") as f:
            f.write(docker_compose)

        print_success("Development docker-compose created")

        # Create development environment file
        dev_env = """
# Development Environment Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8765
MCP_LOG_LEVEL=DEBUG
MCP_LOG_FORMAT=text

# Redis (when using docker-compose)
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=networkx_dev

# Development settings
MAX_GRAPH_SIZE=100000
ENABLE_CACHING=true
CACHE_SIZE_MB=256

# Security (disabled for development)
ENABLE_AUTH=false
API_KEY_REQUIRED=false
RATE_LIMIT_ENABLED=false
        """

        with open(self.project_root / ".env.development", "w") as f:
            f.write(dev_env)

        print_success("Development environment file created")

    def create_development_scripts(self):
        """Create useful development scripts."""
        print_step("Creating development scripts")

        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        # Development server script
        dev_server_script = """#!/usr/bin/env python3
'''Development server with auto-reload and debug features.'''
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Set development environment
os.environ.setdefault("MCP_LOG_LEVEL", "DEBUG")
os.environ.setdefault("MCP_LOG_FORMAT", "text")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# Import and run server
from networkx_mcp.server import main

if __name__ == "__main__":
    main()
"""

        with open(scripts_dir / "dev_server.py", "w") as f:
            f.write(dev_server_script)

        # Make executable
        os.chmod(scripts_dir / "dev_server.py", 0o755)

        # Test runner script
        test_script = """#!/usr/bin/env python3
'''Enhanced test runner with coverage and reporting.'''
import argparse
import subprocess
import sys
from pathlib import Path

def run_tests(args):
    '''Run tests with various options.'''
    project_root = Path(__file__).parent.parent
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if args.unit:
        cmd.append("tests/unit/")
    elif args.integration:
        cmd.append("tests/integration/")
    elif args.property:
        cmd.append("tests/property/")
    elif args.security:
        cmd.append("tests/security/")
    elif args.performance:
        cmd.append("tests/performance/")
    else:
        cmd.append("tests/")
    
    if args.coverage:
        cmd.extend([
            "--cov=src/networkx_mcp",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    if args.verbose:
        cmd.append("-v")
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.fast:
        cmd.extend(["-x", "--tb=short"])
    
    subprocess.run(cmd, cwd=project_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NetworkX MCP Server tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--property", action="store_true", help="Run property-based tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", action="store_true", help="Run tests in parallel")
    parser.add_argument("--fast", "-x", action="store_true", help="Stop on first failure")
    
    args = parser.parse_args()
    run_tests(args)
"""

        with open(scripts_dir / "test_runner.py", "w") as f:
            f.write(test_script)

        os.chmod(scripts_dir / "test_runner.py", 0o755)

        print_success("Development scripts created")

    def validate_setup(self):
        """Validate that the development environment is working."""
        print_step("Validating development environment")

        # Check virtual environment
        if not self.venv_path.exists():
            print_error("Virtual environment not found")
            return False

        # Check dependencies
        if self.is_windows:
            python_executable = self.venv_path / "Scripts" / "python.exe"
        else:
            python_executable = self.venv_path / "bin" / "python"

        try:
            # Test import of main package
            run_command(
                [
                    str(python_executable),
                    "-c",
                    "import networkx_mcp; print('✓ Package import successful')",
                ]
            )

            # Test development tools
            run_command(
                [
                    str(python_executable),
                    "-c",
                    "import pytest; print('✓ pytest available')",
                ]
            )
            run_command(
                [
                    str(python_executable),
                    "-c",
                    "import black; print('✓ black available')",
                ]
            )
            run_command(
                [str(python_executable), "-c", "import ruff; print('✓ ruff available')"]
            )

            print_success("Development environment validation passed")
            return True

        except subprocess.CalledProcessError:
            print_error("Development environment validation failed")
            return False

    def print_next_steps(self):
        """Print next steps for the developer."""
        print(
            f"\n{Colors.BOLD}{Colors.GREEN}🎉 Development Environment Setup Complete!{Colors.END}"
        )

        activation_cmd = (
            "venv\\Scripts\\activate" if self.is_windows else "source venv/bin/activate"
        )

        next_steps = f"""
{Colors.BOLD}Next Steps:{Colors.END}

1. {Colors.BLUE}Activate virtual environment:{Colors.END}
   {activation_cmd}

2. {Colors.BLUE}Start development server:{Colors.END}
   python scripts/dev_server.py

3. {Colors.BLUE}Run tests:{Colors.END}
   python scripts/test_runner.py --coverage

4. {Colors.BLUE}Start Redis (optional):{Colors.END}
   docker-compose -f docker-compose.dev.yml up -d

5. {Colors.BLUE}Open in VSCode:{Colors.END}
   code .

{Colors.BOLD}Useful Commands:{Colors.END}
- Run linting: ruff check src tests
- Format code: black src tests
- Type checking: mypy src/networkx_mcp
- Run benchmarks: python -m asv run
- Documentation: mkdocs serve

{Colors.BOLD}Development URLs:{Colors.END}
- Server: http://localhost:8765
- Redis UI: http://localhost:8081 (if using docker-compose)
- Documentation: http://localhost:8000 (when running mkdocs serve)

{Colors.YELLOW}Need help? Check docs/development/ or open an issue!{Colors.END}
        """

        print(next_steps)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up NetworkX MCP Server development environment"
    )
    parser.add_argument(
        "--skip-venv", action="store_true", help="Skip virtual environment creation"
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-hooks", action="store_true", help="Skip pre-commit hooks"
    )
    parser.add_argument(
        "--skip-ide", action="store_true", help="Skip IDE configuration"
    )
    parser.add_argument(
        "--skip-services", action="store_true", help="Skip development services"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing setup"
    )

    args = parser.parse_args()

    # Find project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    print(
        f"{Colors.BOLD}🕸️ NetworkX MCP Server - Development Environment Setup{Colors.END}"
    )
    print(f"Project root: {project_root}")

    setup = DevEnvironmentSetup(project_root)

    if args.validate_only:
        success = setup.validate_setup()
        sys.exit(0 if success else 1)

    try:
        # Always check prerequisites
        setup.check_prerequisites()

        # Run setup steps
        if not args.skip_venv:
            setup.create_virtual_environment()

        if not args.skip_deps:
            setup.install_dependencies()

        if not args.skip_hooks:
            setup.setup_pre_commit_hooks()

        if not args.skip_ide:
            setup.create_ide_config()

        if not args.skip_services:
            setup.setup_development_services()

        # Always create development scripts
        setup.create_development_scripts()

        # Validate setup
        if setup.validate_setup():
            setup.print_next_steps()
        else:
            print_error(
                "Setup validation failed. Please check the error messages above."
            )
            sys.exit(1)

    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
