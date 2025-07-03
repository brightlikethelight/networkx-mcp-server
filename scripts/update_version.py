#!/usr/bin/env python3
"""Version update script for NetworkX MCP Server.

This script updates version numbers across all project files
for automated release management.
"""

import argparse
import re
import sys
from pathlib import Path

import toml
import yaml


class VersionUpdater:
    """Handles version updates across project files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.files_to_update: list[tuple[Path, str, str]] = []

    def update_pyproject_toml(self, new_version: str) -> bool:
        """Update version in pyproject.toml."""
        pyproject_file = self.project_root / "pyproject.toml"

        if not pyproject_file.exists():
            print(f"Warning: {pyproject_file} not found")
            return False

        try:
            with open(pyproject_file) as f:
                data = toml.load(f)

            old_version = data.get("project", {}).get("version", "unknown")
            data["project"]["version"] = new_version

            with open(pyproject_file, "w") as f:
                toml.dump(data, f)

            self.files_to_update.append((pyproject_file, old_version, new_version))
            print(f"✅ Updated {pyproject_file}: {old_version} → {new_version}")
            return True

        except Exception as e:
            print(f"❌ Failed to update {pyproject_file}: {e}")
            return False

    def update_version_py(self, new_version: str) -> bool:
        """Update version in __version__.py file."""
        version_file = self.project_root / "src" / "networkx_mcp" / "__version__.py"

        if not version_file.exists():
            print(f"Warning: {version_file} not found")
            return False

        try:
            with open(version_file) as f:
                content = f.read()

            # Extract current version
            version_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'
            match = re.search(version_pattern, content)
            old_version = match.group(1) if match else "unknown"

            # Update version
            new_content = re.sub(
                version_pattern, f'__version__ = "{new_version}"', content
            )

            with open(version_file, "w") as f:
                f.write(new_content)

            self.files_to_update.append((version_file, old_version, new_version))
            print(f"✅ Updated {version_file}: {old_version} → {new_version}")
            return True

        except Exception as e:
            print(f"❌ Failed to update {version_file}: {e}")
            return False

    def update_helm_chart(self, new_version: str) -> bool:
        """Update version in Helm Chart.yaml."""
        chart_file = self.project_root / "helm" / "networkx-mcp" / "Chart.yaml"

        if not chart_file.exists():
            print(f"Warning: {chart_file} not found")
            return False

        try:
            with open(chart_file) as f:
                data = yaml.safe_load(f)

            old_version = data.get("version", "unknown")
            old_app_version = data.get("appVersion", "unknown")

            data["version"] = new_version
            data["appVersion"] = new_version

            with open(chart_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            self.files_to_update.append(
                (chart_file, f"{old_version}/{old_app_version}", new_version)
            )
            print(
                f"✅ Updated {chart_file}: {old_version}/{old_app_version} → {new_version}"
            )
            return True

        except Exception as e:
            print(f"❌ Failed to update {chart_file}: {e}")
            return False

    def update_docker_compose(self, new_version: str) -> bool:
        """Update version in docker-compose.yml."""
        compose_file = self.project_root / "docker-compose.yml"

        if not compose_file.exists():
            print(f"Warning: {compose_file} not found")
            return False

        try:
            with open(compose_file) as f:
                content = f.read()

            # Update image tag references
            old_content = content
            content = re.sub(
                r"(image:\s*networkx-mcp:)[^-\n]*", f"\\g<1>{new_version}", content
            )

            # Update BUILD_DATE and VERSION args
            content = re.sub(r"(VERSION:\s*)[^\n]*", f"\\g<1>{new_version}", content)

            if content != old_content:
                with open(compose_file, "w") as f:
                    f.write(content)

                self.files_to_update.append((compose_file, "various", new_version))
                print(f"✅ Updated {compose_file}: version references → {new_version}")

            return True

        except Exception as e:
            print(f"❌ Failed to update {compose_file}: {e}")
            return False

    def update_kubernetes_manifests(self, new_version: str) -> bool:
        """Update version in Kubernetes manifests."""
        k8s_dir = self.project_root / "k8s"

        if not k8s_dir.exists():
            print(f"Warning: {k8s_dir} not found")
            return False

        success = True

        for manifest_file in k8s_dir.glob("*.yaml"):
            try:
                with open(manifest_file) as f:
                    content = f.read()

                old_content = content

                # Update image references
                content = re.sub(
                    r"(image:\s*networkx-mcp:)[^\n]*", f"\\g<1>{new_version}", content
                )

                # Update version labels
                content = re.sub(
                    r'(version:\s*)["\']?[^"\'\n]*["\']?',
                    f'\\g<1>"{new_version}"',
                    content,
                )

                if content != old_content:
                    with open(manifest_file, "w") as f:
                        f.write(content)

                    self.files_to_update.append((manifest_file, "various", new_version))
                    print(
                        f"✅ Updated {manifest_file}: version references → {new_version}"
                    )

            except Exception as e:
                print(f"❌ Failed to update {manifest_file}: {e}")
                success = False

        return success

    def update_readme(self, new_version: str) -> bool:
        """Update version references in README."""
        readme_file = self.project_root / "README.md"

        if not readme_file.exists():
            print(f"Warning: {readme_file} not found")
            return False

        try:
            with open(readme_file) as f:
                content = f.read()

            old_content = content

            # Update version badges and references
            content = re.sub(
                r"(networkx-mcp:)[v]?[0-9]+\.[0-9]+\.[0-9]+[^\s\]]*",
                f"\\g<1>v{new_version}",
                content,
            )

            # Update pip install commands
            content = re.sub(
                r"(pip install networkx-mcp==)[0-9]+\.[0-9]+\.[0-9]+[^\s]*",
                f"\\g<1>{new_version}",
                content,
            )

            if content != old_content:
                with open(readme_file, "w") as f:
                    f.write(content)

                self.files_to_update.append((readme_file, "various", new_version))
                print(f"✅ Updated {readme_file}: version references → {new_version}")

            return True

        except Exception as e:
            print(f"❌ Failed to update {readme_file}: {e}")
            return False

    def update_all_files(self, new_version: str) -> bool:
        """Update version in all project files."""
        print(f"🔄 Updating version to {new_version}")

        success = True

        # Update core files
        success &= self.update_pyproject_toml(new_version)
        success &= self.update_version_py(new_version)

        # Update deployment files
        success &= self.update_helm_chart(new_version)
        success &= self.update_docker_compose(new_version)
        success &= self.update_kubernetes_manifests(new_version)

        # Update documentation
        success &= self.update_readme(new_version)

        return success

    def validate_version_format(self, version: str) -> bool:
        """Validate semantic version format."""
        semver_pattern = r"^([0-9]+)\.([0-9]+)\.([0-9]+)(?:-([a-zA-Z0-9\-\.]+))?(?:\+([a-zA-Z0-9\-\.]+))?$"

        if not re.match(semver_pattern, version):
            print(f"❌ Invalid semantic version format: {version}")
            print("   Expected format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]")
            return False

        return True

    def get_current_version(self) -> str:
        """Get current version from pyproject.toml."""
        pyproject_file = self.project_root / "pyproject.toml"

        if not pyproject_file.exists():
            return "0.0.0"

        try:
            with open(pyproject_file) as f:
                data = toml.load(f)

            return data.get("project", {}).get("version", "0.0.0")

        except Exception:
            return "0.0.0"

    def create_summary(self) -> None:
        """Create update summary."""
        if not self.files_to_update:
            print("ℹ️  No files were updated")
            return

        print("\n📋 Update Summary:")
        print(f"   Files updated: {len(self.files_to_update)}")

        for file_path, old_version, new_version in self.files_to_update:
            rel_path = file_path.relative_to(self.project_root)
            print(f"   • {rel_path}: {old_version} → {new_version}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update version numbers across project files"
    )
    parser.add_argument(
        "version", help="New version number (semantic versioning format)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate version format"
    )

    args = parser.parse_args()

    # Initialize updater
    updater = VersionUpdater(args.project_root)

    # Validate version format
    if not updater.validate_version_format(args.version):
        sys.exit(1)

    if args.validate_only:
        print(f"✅ Version format is valid: {args.version}")
        sys.exit(0)

    # Get current version
    current_version = updater.get_current_version()
    print(f"📌 Current version: {current_version}")
    print(f"🎯 Target version: {args.version}")

    if args.dry_run:
        print("🔍 DRY RUN: No files will be modified")
        # TODO: Implement dry run functionality
        sys.exit(0)

    # Update all files
    success = updater.update_all_files(args.version)

    # Create summary
    updater.create_summary()

    if success:
        print(f"\n🎉 Successfully updated version to {args.version}")
        sys.exit(0)
    else:
        print("\n❌ Some updates failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
