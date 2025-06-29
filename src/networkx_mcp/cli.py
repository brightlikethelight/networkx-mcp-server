"""CLI interface for NetworkX MCP server testing and debugging."""

import argparse
import asyncio

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from networkx_mcp.core.algorithms import GraphAlgorithms
from networkx_mcp.core.graph_operations import GraphManager
from networkx_mcp.core.io_handlers import GraphIOHandler
from networkx_mcp.utils.monitoring import OperationCounter
from networkx_mcp.utils.monitoring import PerformanceMonitor


console = Console()


class NetworkXCLI:
    """Interactive CLI for NetworkX MCP server."""

    def __init__(self):
        self.graph_manager = GraphManager()
        self.performance_monitor = PerformanceMonitor()
        self.operation_counter = OperationCounter()
        self.current_graph = None

    def print_banner(self):
        """Print CLI banner."""
        banner = """
╔═══════════════════════════════════════════════════════════╗
║          NetworkX MCP Server - Interactive CLI            ║
║                                                           ║
║  Commands: help, create, list, info, add, analyze, exit  ║
╚═══════════════════════════════════════════════════════════╝
        """
        console.print(banner, style="bold blue")

    def print_help(self):
        """Print help information."""
        help_text = """
[bold]Available Commands:[/bold]

[green]Graph Management:[/green]
  create <graph_id> [type]     - Create a new graph
  list                         - List all graphs
  info [graph_id]              - Show graph information
  delete <graph_id>            - Delete a graph
  select <graph_id>            - Select active graph
  clear <graph_id>             - Clear graph contents

[green]Graph Building:[/green]
  add nodes <n1> <n2> ...      - Add nodes
  add edge <source> <target>   - Add edge
  import <format> <file>       - Import graph from file
  export <format> <file>       - Export graph to file

[green]Analysis:[/green]
  analyze centrality [type]    - Calculate centrality
  analyze path <src> <dst>     - Find shortest path
  analyze components           - Find connected components
  analyze metrics              - Calculate graph metrics
  analyze communities          - Detect communities

[green]Other:[/green]
  monitor                      - Show performance statistics
  benchmark <size>             - Run performance benchmark
  demo <example>               - Run demo (social/transport/citation)
  help                         - Show this help
  exit                         - Exit CLI
        """
        console.print(help_text)

    async def create_graph(self, args: list):
        """Create a new graph."""
        if len(args) < 1:
            console.print("[red]Usage: create <graph_id> [type][/red]")
            return

        graph_id = args[0]
        graph_type = args[1] if len(args) > 1 else "Graph"

        try:
            self.graph_manager.create_graph(graph_id, graph_type)
            console.print(f"[green]✓ Created graph '{graph_id}' (type: {graph_type})[/green]")
            self.current_graph = graph_id
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    def list_graphs(self):
        """List all graphs."""
        graphs = self.graph_manager.list_graphs()

        if not graphs:
            console.print("[yellow]No graphs created yet[/yellow]")
            return

        table = Table(title="Available Graphs")
        table.add_column("Graph ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Nodes", justify="right")
        table.add_column("Edges", justify="right")
        table.add_column("Created", style="green")

        for graph in graphs:
            table.add_row(
                graph["graph_id"],
                graph["graph_type"],
                str(graph["num_nodes"]),
                str(graph["num_edges"]),
                graph["metadata"]["created_at"][:19]
            )

        console.print(table)

        if self.current_graph:
            console.print(f"\n[bold]Current graph: {self.current_graph}[/bold]")

    def show_graph_info(self, graph_id: Optional[str] = None):
        """Show detailed graph information."""
        if not graph_id:
            graph_id = self.current_graph

        if not graph_id:
            console.print("[red]No graph selected. Use 'select <graph_id>'[/red]")
            return

        try:
            info = self.graph_manager.get_graph_info(graph_id)

            # Create info panel
            info_text = f"""
[bold]Graph: {graph_id}[/bold]
Type: {info['graph_type']}
Nodes: {info['num_nodes']}
Edges: {info['num_edges']}
Density: {info['density']:.4f}
Directed: {info['is_directed']}
MultiGraph: {info['is_multigraph']}
            """

            if "degree_stats" in info:
                stats = info["degree_stats"]
                info_text += f"""
Degree Stats:
  Average: {stats['average']:.2f}
  Min: {stats['min']}
  Max: {stats['max']}
            """

            console.print(Panel(info_text, title="Graph Information"))

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    async def add_nodes(self, args: list):
        """Add nodes to current graph."""
        if not self.current_graph:
            console.print("[red]No graph selected. Use 'select <graph_id>'[/red]")
            return

        if not args:
            console.print("[red]Usage: add nodes <node1> <node2> ...[/red]")
            return

        try:
            result = self.graph_manager.add_nodes_from(self.current_graph, args)
            console.print(f"[green]✓ Added {result['nodes_added']} nodes[/green]")
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    async def add_edge(self, args: list):
        """Add edge to current graph."""
        if not self.current_graph:
            console.print("[red]No graph selected. Use 'select <graph_id>'[/red]")
            return

        if len(args) < 2:
            console.print("[red]Usage: add edge <source> <target> [weight][/red]")
            return

        source, target = args[0], args[1]
        attrs = {"weight": float(args[2])} if len(args) > 2 else {}

        try:
            self.graph_manager.add_edge(self.current_graph, source, target, **attrs)
            console.print(f"[green]✓ Added edge {source} -> {target}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    async def analyze(self, args: list):
        """Run various analyses."""
        if not args:
            console.print("[red]Usage: analyze <type> [options][/red]")
            return

        if not self.current_graph:
            console.print("[red]No graph selected. Use 'select <graph_id>'[/red]")
            return

        analysis_type = args[0]
        graph = self.graph_manager.get_graph(self.current_graph)

        try:
            if analysis_type == "centrality":
                measure = args[1] if len(args) > 1 else "degree"
                result = GraphAlgorithms.centrality_measures(graph, [measure])

                # Display top nodes
                centrality = result.get(f"{measure}_centrality", {})
                if centrality:
                    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                    table = Table(title=f"{measure.title()} Centrality")
                    table.add_column("Node", style="cyan")
                    table.add_column("Score", justify="right")

                    for node, score in sorted_nodes:
                        table.add_row(str(node), f"{score:.4f}")

                    console.print(table)

            elif analysis_type == "path":
                if len(args) < 3:
                    console.print("[red]Usage: analyze path <source> <target>[/red]")
                    return

                source, target = args[1], args[2]
                result = GraphAlgorithms.shortest_path(graph, source, target)

                if result.get("path"):
                    path = " → ".join(map(str, result["path"]))
                    console.print(f"[green]Shortest path: {path}[/green]")
                    console.print(f"Length: {result.get('length', 'N/A')}")
                else:
                    console.print(f"[yellow]No path from {source} to {target}[/yellow]")

            elif analysis_type == "components":
                result = GraphAlgorithms.connected_components(graph)
                console.print(f"[green]Connected components: {result['num_components']}[/green]")
                console.print(f"Is connected: {result.get('is_connected', 'N/A')}")

                if result["num_components"] > 1:
                    sizes = [len(comp) for comp in result["connected_components"]]
                    console.print(f"Component sizes: {sizes[:10]}...")

            elif analysis_type == "metrics":
                stats = GraphAlgorithms.graph_statistics(graph)

                metrics_text = f"""
Nodes: {stats['num_nodes']}
Edges: {stats['num_edges']}
Density: {stats['density']:.4f}
Avg Degree: {stats['degree_stats']['mean']:.2f}
Max Degree: {stats['degree_stats']['max']}
Min Degree: {stats['degree_stats']['min']}
                """

                if "diameter" in stats:
                    metrics_text += f"""
Diameter: {stats['diameter']}
Radius: {stats['radius']}
                    """

                console.print(Panel(metrics_text, title="Graph Metrics"))

            elif analysis_type == "communities":
                result = GraphAlgorithms.community_detection(graph)
                console.print(f"[green]Communities found: {result['num_communities']}[/green]")
                console.print(f"Modularity: {result.get('modularity', 'N/A'):.4f}")

                # Show community sizes
                sizes = result.get("community_sizes", [])
                if sizes:
                    console.print(f"Community sizes: {sizes[:10]}...")

            else:
                console.print(f"[red]Unknown analysis type: {analysis_type}[/red]")

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")

    def show_monitor_stats(self):
        """Show performance monitoring statistics."""
        perf_stats = self.performance_monitor.get_statistics()
        op_counts = self.operation_counter.get_counts()

        # Performance table
        perf_table = Table(title="Performance Statistics")
        perf_table.add_column("Operation", style="cyan")
        perf_table.add_column("Count", justify="right")
        perf_table.add_column("Avg (ms)", justify="right")
        perf_table.add_column("Total (ms)", justify="right")

        for op, stats in perf_stats.get("operations", {}).items():
            perf_table.add_row(
                op,
                str(stats["count"]),
                f"{stats['mean_ms']:.2f}",
                f"{stats['total_ms']:.2f}"
            )

        console.print(perf_table)

        # Operation counts
        console.print(f"\n[bold]Total Operations:[/bold] {op_counts['total_operations']}")
        console.print(f"[bold]Error Rate:[/bold] {op_counts['error_rate']:.2f}%")
        console.print(f"[bold]Uptime:[/bold] {op_counts['uptime']}")

    async def run_benchmark(self, size: int = 100):
        """Run performance benchmark."""
        console.print(f"[yellow]Running benchmark with {size} nodes...[/yellow]")

        # Create test graph
        test_id = f"benchmark_{size}"
        self.graph_manager.create_graph(test_id, "Graph")

        # Add nodes
        import time
        start = time.time()
        nodes = list(range(size))
        self.graph_manager.add_nodes_from(test_id, nodes)
        node_time = time.time() - start

        # Add random edges
        import random  # Using for non-cryptographic test data generation only
        edges = []
        for _ in range(size * 2):
            u, v = random.sample(nodes, 2)
            edges.append((u, v))

        start = time.time()
        self.graph_manager.add_edges_from(test_id, edges)
        edge_time = time.time() - start

        # Run algorithms
        graph = self.graph_manager.get_graph(test_id)

        start = time.time()
        GraphAlgorithms.centrality_measures(graph, ["degree"])
        centrality_time = time.time() - start

        start = time.time()
        GraphAlgorithms.connected_components(graph)
        component_time = time.time() - start

        # Clean up
        self.graph_manager.delete_graph(test_id)

        # Show results
        results = f"""
[bold]Benchmark Results ({size} nodes, {len(edges)} edges):[/bold]
Node creation: {node_time*1000:.2f} ms
Edge creation: {edge_time*1000:.2f} ms
Centrality calculation: {centrality_time*1000:.2f} ms
Component detection: {component_time*1000:.2f} ms
Total: {(node_time + edge_time + centrality_time + component_time)*1000:.2f} ms
        """
        console.print(Panel(results, title="Benchmark Complete"))

    async def run_demo(self, demo_type: str):
        """Run demonstration."""
        console.print(f"[yellow]Running {demo_type} demonstration...[/yellow]")

        if demo_type == "social":
            # Create small social network
            self.graph_manager.create_graph("demo_social", "Graph")
            self.current_graph = "demo_social"

            # Add people
            people = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
            self.graph_manager.add_nodes_from("demo_social", people)

            # Add friendships
            friendships = [
                ("Alice", "Bob"), ("Alice", "Charlie"),
                ("Bob", "Diana"), ("Charlie", "Diana"),
                ("Diana", "Eve")
            ]
            self.graph_manager.add_edges_from("demo_social", friendships)

            console.print("[green]✓ Created social network demo[/green]")
            self.show_graph_info("demo_social")

        elif demo_type == "transport":
            # Create transport network
            self.graph_manager.create_graph("demo_transport", "DiGraph")
            self.current_graph = "demo_transport"

            # Add stations
            stations = ["A", "B", "C", "D", "E"]
            self.graph_manager.add_nodes_from("demo_transport", stations)

            # Add routes with times
            routes = [
                ("A", "B", {"time": 5}), ("B", "C", {"time": 3}),
                ("A", "D", {"time": 10}), ("D", "C", {"time": 2}),
                ("B", "E", {"time": 4}), ("C", "E", {"time": 3})
            ]
            self.graph_manager.add_edges_from("demo_transport", routes)

            console.print("[green]✓ Created transport network demo[/green]")
            self.show_graph_info("demo_transport")

        else:
            console.print(f"[red]Unknown demo: {demo_type}[/red]")

    async def interactive_mode(self):
        """Run interactive CLI mode."""
        self.print_banner()

        while True:
            try:
                # Show prompt
                prompt = f"[{self.current_graph or 'no graph'}]> " if self.current_graph else "> "
                command = console.input(prompt).strip()

                if not command:
                    continue

                parts = command.split()
                cmd = parts[0].lower()
                args = parts[1:]

                if cmd in {"exit", "quit"}:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                elif cmd == "help":
                    self.print_help()

                elif cmd == "create":
                    await self.create_graph(args)

                elif cmd == "list":
                    self.list_graphs()

                elif cmd == "info":
                    graph_id = args[0] if args else None
                    self.show_graph_info(graph_id)

                elif cmd == "select":
                    if args:
                        self.current_graph = args[0]
                        console.print(f"[green]Selected graph: {self.current_graph}[/green]")
                    else:
                        console.print("[red]Usage: select <graph_id>[/red]")

                elif cmd == "delete":
                    if args:
                        try:
                            self.graph_manager.delete_graph(args[0])
                            console.print(f"[green]✓ Deleted graph '{args[0]}'[/green]")
                            if self.current_graph == args[0]:
                                self.current_graph = None
                        except Exception as e:
                            console.print(f"[red]✗ Error: {e}[/red]")
                    else:
                        console.print("[red]Usage: delete <graph_id>[/red]")

                elif cmd == "clear":
                    graph_id = args[0] if args else self.current_graph
                    if graph_id:
                        try:
                            self.graph_manager.clear_graph(graph_id)
                            console.print(f"[green]✓ Cleared graph '{graph_id}'[/green]")
                        except Exception as e:
                            console.print(f"[red]✗ Error: {e}[/red]")
                    else:
                        console.print("[red]No graph specified[/red]")

                elif cmd == "add":
                    if args and args[0] == "nodes":
                        await self.add_nodes(args[1:])
                    elif args and args[0] == "edge":
                        await self.add_edge(args[1:])
                    else:
                        console.print("[red]Usage: add nodes|edge ...[/red]")

                elif cmd == "analyze":
                    await self.analyze(args)

                elif cmd == "monitor":
                    self.show_monitor_stats()

                elif cmd == "benchmark":
                    size = int(args[0]) if args else 100
                    await self.run_benchmark(size)

                elif cmd == "demo":
                    if args:
                        await self.run_demo(args[0])
                    else:
                        console.print("[red]Usage: demo social|transport[/red]")

                elif cmd == "import":
                    if len(args) >= 2:
                        format_type, filepath = args[0], args[1]
                        try:
                            graph = GraphIOHandler.import_graph(format_type, path=filepath)
                            graph_id = Path(filepath).stem
                            self.graph_manager.graphs[graph_id] = graph
                            self.graph_manager.metadata[graph_id] = {
                                "created_at": "imported",
                                "graph_type": type(graph).__name__
                            }
                            self.current_graph = graph_id
                            console.print(f"[green]✓ Imported graph as '{graph_id}'[/green]")
                        except Exception as e:
                            console.print(f"[red]✗ Error: {e}[/red]")
                    else:
                        console.print("[red]Usage: import <format> <file>[/red]")

                elif cmd == "export":
                    if len(args) >= 2 and self.current_graph:
                        format_type, filepath = args[0], args[1]
                        try:
                            graph = self.graph_manager.get_graph(self.current_graph)
                            GraphIOHandler.export_graph(graph, format_type, filepath)
                            console.print(f"[green]✓ Exported to {filepath}[/green]")
                        except Exception as e:
                            console.print(f"[red]✗ Error: {e}[/red]")
                    else:
                        console.print("[red]Usage: export <format> <file>[/red]")

                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
                    console.print("Type 'help' for available commands")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="NetworkX MCP Server CLI")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode")
    parser.add_argument("--benchmark", type=int, help="Run benchmark with N nodes")
    parser.add_argument("--demo", choices=["social", "transport", "citation"],
                       help="Run demonstration")

    args = parser.parse_args()

    cli = NetworkXCLI()

    if args.benchmark:
        asyncio.run(cli.run_benchmark(args.benchmark))
    elif args.demo:
        asyncio.run(cli.run_demo(args.demo))
    else:
        # Interactive mode
        asyncio.run(cli.interactive_mode())


if __name__ == "__main__":
    main()
