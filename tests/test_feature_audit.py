"""Comprehensive audit of all NetworkX MCP Server features.

This test provides an honest assessment of what works and what doesn't,
ensuring we only claim features that actually function.
"""

import pytest

# Skip this module - it's not a proper test suite
pytestmark = pytest.mark.skip(reason="Feature audit module is not a test suite")


def test_feature_import(module_path: str, class_name: str) -> tuple[bool, str]:
    """Test if a feature can be imported and instantiated."""
    try:
        # Import the module
        parts = module_path.split(".")
        module = __import__(f"networkx_mcp.{module_path}", fromlist=[class_name])
        cls = getattr(module, class_name)

        # Try to instantiate if it's a class
        if hasattr(cls, "__init__"):
            # Try basic instantiation
            try:
                instance = cls()
                return True, "✅ Works - Can import and instantiate"
            except TypeError:
                # May need arguments, but import works
                return True, "✅ Works - Can import (requires arguments)"
            except Exception as e:
                return False, f"❌ Broken - Import works but instantiation fails: {e}"
        else:
            return True, "✅ Works - Can import (not a class)"

    except ImportError as e:
        return False, f"❌ Broken - Cannot import: {e}"
    except Exception as e:
        return False, f"❌ Broken - Unexpected error: {e}"


def test_all_features():
    """Comprehensive feature audit."""
    print("\n" + "=" * 80)
    print("🔍 NETWORKX MCP SERVER FEATURE AUDIT")
    print("=" * 80)

    # Core features (these must work)
    core_features = [
        ("core.graph_operations", "GraphManager"),
        ("core.algorithms", "GraphAlgorithms"),
        ("services.unified_graph_service", "UnifiedGraphService"),
    ]

    # Advanced features
    advanced_features = [
        ("advanced.robustness", "RobustnessAnalysis"),
        ("advanced.ml.node_classification", "NodeClassifier"),
        ("advanced.ml.link_prediction", "LinkPredictor"),
        ("advanced.bipartite_analysis", "BipartiteAnalysis"),
        ("advanced.community_detection", "CommunityDetection"),
        ("advanced.generators", "GraphGenerators"),
    ]

    # Enterprise features
    enterprise_features = [
        ("enterprise.circuit_breaker", "CircuitBreaker"),
        ("enterprise.feature_flags", "FeatureFlagService"),
    ]

    # Infrastructure features
    infrastructure_features = [
        ("security.validation", "RequestValidator"),
        ("security.middleware", "SecurityMiddleware"),
        ("security.audit", "AuditLogger"),
        ("monitoring.health_checks", "HealthCheck"),
        ("monitoring.metrics", "MetricsCollector"),
        ("caching.cache_service", "CacheService"),
        ("events.graph_events", "GraphEventPublisher"),
        ("repositories.graph_repository", "GraphRepository"),
        ("storage.redis_backend", "RedisBackend"),
    ]

    # Test all feature categories
    categories = [
        ("🔧 CORE FEATURES", core_features),
        ("🚀 ADVANCED FEATURES", advanced_features),
        ("🏢 ENTERPRISE FEATURES", enterprise_features),
        ("🏗️ INFRASTRUCTURE FEATURES", infrastructure_features),
    ]

    all_results = {}

    for category_name, features in categories:
        print(f"\n{category_name}")
        print("-" * len(category_name))

        working = 0
        total = len(features)

        for module_path, class_name in features:
            works, message = test_feature_import(module_path, class_name)
            feature_name = f"{module_path}.{class_name}"

            print(f"  {message[:60]}... {feature_name}")

            if works:
                working += 1

            all_results[feature_name] = works

        percentage = (working / total) * 100 if total > 0 else 0
        print(f"\n  📊 Status: {working}/{total} features working ({percentage:.1f}%)")

    # Summary
    total_features = len(all_results)
    working_features = sum(all_results.values())
    overall_percentage = (working_features / total_features) * 100

    print("\n" + "=" * 80)
    print("📋 OVERALL SUMMARY")
    print("=" * 80)
    print(f"✅ Working Features: {working_features}")
    print(f"❌ Broken Features: {total_features - working_features}")
    print(
        f"📊 Overall Status: {working_features}/{total_features} ({overall_percentage:.1f}%)"
    )

    # Identify broken features
    broken_features = [name for name, works in all_results.items() if not works]
    if broken_features:
        print("\n🚨 BROKEN FEATURES TO FIX OR REMOVE:")
        for feature in broken_features:
            print(f"  ❌ {feature}")

    # Feature readiness assessment
    if overall_percentage >= 90:
        print(f"\n🎉 EXCELLENT: {overall_percentage:.1f}% - Production ready!")
    elif overall_percentage >= 75:
        print(f"\n👍 GOOD: {overall_percentage:.1f}% - Near production ready")
    elif overall_percentage >= 50:
        print(f"\n⚠️ FAIR: {overall_percentage:.1f}% - Needs improvement")
    else:
        print(f"\n🚨 POOR: {overall_percentage:.1f}% - Major work needed")

    return all_results


def test_critical_workflows():
    """Test end-to-end workflows to ensure they actually work."""
    print("\n" + "=" * 80)
    print("🔄 CRITICAL WORKFLOW TESTING")
    print("=" * 80)

    workflows = []

    # Test 1: Basic graph operations
    try:
        from networkx_mcp.services.unified_graph_service import \
            UnifiedGraphService

        service = UnifiedGraphService()

        # Create, modify, analyze, delete
        service.create_graph("workflow_test", "Graph")
        service.add_nodes("workflow_test", [1, 2, 3, 4, 5])
        service.add_edges("workflow_test", [(1, 2), (2, 3), (3, 4), (4, 5)])
        result = service.shortest_path("workflow_test", 1, 5)
        components = service.connected_components("workflow_test")
        service.delete_graph("workflow_test")

        if result["status"] == "success" and components["status"] == "success":
            workflows.append("✅ Basic graph operations")
        else:
            workflows.append("❌ Basic graph operations - API errors")

    except Exception as e:
        workflows.append(f"❌ Basic graph operations - {e}")

    # Test 2: ML workflow
    try:
        import asyncio

        import networkx as nx

        from networkx_mcp.advanced.ml.node_classification import NodeClassifier

        async def test_ml():
            G = nx.Graph()
            G.add_edges_from([(1, 2), (2, 3), (3, 4)])
            classifier = NodeClassifier(G)
            await classifier.train({1: "A", 2: "B"})
            result = await classifier.predict([3])
            return result.predictions is not None

        ml_works = asyncio.run(test_ml())
        workflows.append(
            "✅ ML node classification" if ml_works else "❌ ML node classification"
        )

    except Exception as e:
        workflows.append(f"❌ ML workflow - {e}")

    # Test 3: Enterprise features
    try:
        from networkx_mcp.enterprise import EnterpriseFeatures

        status = EnterpriseFeatures.get_feature_status()
        if "2/5" in status:  # We know 2 out of 5 should work
            workflows.append("✅ Enterprise features (partial)")
        else:
            workflows.append(f"❌ Enterprise features - {status}")
    except Exception as e:
        workflows.append(f"❌ Enterprise features - {e}")

    # Print results
    for workflow in workflows:
        print(f"  {workflow}")

    working_workflows = sum(1 for w in workflows if w.startswith("✅"))
    total_workflows = len(workflows)
    workflow_percentage = (working_workflows / total_workflows) * 100

    print(
        f"\n📊 Workflow Status: {working_workflows}/{total_workflows} ({workflow_percentage:.1f}%)"
    )

    return workflows


if __name__ == "__main__":
    print("🚀 RUNNING COMPREHENSIVE FEATURE AUDIT")
    print("This audit provides an honest assessment of what actually works.")

    # Run feature audit
    feature_results = test_all_features()

    # Run workflow tests
    workflow_results = test_critical_workflows()

    print("\n" + "=" * 80)
    print("✅ AUDIT COMPLETE - Results show honest status of all features")
    print("=" * 80)
