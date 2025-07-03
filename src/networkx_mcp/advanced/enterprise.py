"""Enterprise features for production deployments."""

import base64
import hashlib
import io
import json
import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

try:
    import schedule

    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False
    schedule = None

try:
    from jinja2 import Template

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Template = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table)

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    letter = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    inch = None
    Image = None
    PageBreak = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None

# Performance thresholds
MAX_NODES_FOR_EXPENSIVE_METRICS = 1000
DISPLAY_LIMIT_FOR_ITEMS = 5


# Optional imports
try:
    import community as community_louvain

    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    community_louvain = None


logger = logging.getLogger(__name__)


class EnterpriseFeatures:
    """Enterprise-grade features for production deployments."""

    def __init__(self, cache_dir: str = "./cache", max_workers: int | None = None):
        """Initialize enterprise features with caching and worker pool."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers or mp.cpu_count()
        self._cache = {}
        self._scheduler_thread = None
        self._scheduled_jobs = []

    def batch_analysis(
        self,
        graphs: list[tuple[str, nx.Graph]],
        operations: list[dict[str, Any]],
        parallel: bool = True,
        batch_size: int = 10,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """
        Process multiple graphs in parallel with batching.

        Parameters:
        -----------
        graphs : list
            List of (graph_id, graph) tuples
        operations : list
            List of operations to perform on each graph
        parallel : bool
            Use parallel processing
        batch_size : int
            Batch size for processing
        progress_callback : callable
            Progress notification callback

        Returns:
        --------
        Dict containing results for all graphs
        """
        start_time = time.time()
        results = {}
        total_graphs = len(graphs)

        if parallel and len(graphs) > 1:
            # Process in parallel batches
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                for i in range(0, len(graphs), batch_size):
                    batch = graphs[i : i + batch_size]
                    batch_futures = []

                    for graph_id, graph in batch:
                        future = executor.submit(
                            self._process_single_graph, graph_id, graph, operations
                        )
                        batch_futures.append((graph_id, future))

                    # Collect batch results
                    for graph_id, future in batch_futures:
                        try:
                            results[graph_id] = future.result(timeout=300)
                        except Exception as e:
                            logger.error(f"Error processing {graph_id}: {e}")
                            results[graph_id] = {"error": str(e)}

                    # Progress callback
                    if progress_callback:
                        progress = min(i + batch_size, total_graphs) / total_graphs
                        progress_callback(
                            progress,
                            f"Processed {min(i + batch_size, total_graphs)}/{total_graphs} graphs",
                        )
        else:
            # Sequential processing
            for i, (graph_id, graph) in enumerate(graphs):
                results[graph_id] = self._process_single_graph(
                    graph_id, graph, operations
                )

                if progress_callback:
                    progress_callback(
                        (i + 1) / total_graphs,
                        f"Processed {i + 1}/{total_graphs} graphs",
                    )

        total_time = time.time() - start_time

        return {
            "results": results,
            "summary": {
                "total_graphs": total_graphs,
                "successful": sum(1 for r in results.values() if "error" not in r),
                "failed": sum(1 for r in results.values() if "error" in r),
                "total_time": total_time,
                "avg_time_per_graph": total_time / total_graphs,
            },
            "batch_config": {
                "parallel": parallel,
                "batch_size": batch_size,
                "max_workers": self.max_workers,
            },
        }

    def analysis_workflow(
        self,
        graph: nx.Graph,
        workflow_config: list[dict[str, Any]],
        cache_intermediate: bool = True,
    ) -> dict[str, Any]:
        """
        Chain operations with intermediate caching.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        workflow_config : list
            List of workflow steps
        cache_intermediate : bool
            Cache intermediate results

        Returns:
        --------
        Dict containing workflow results
        """
        start_time = time.time()
        results = {}
        current_graph = graph.copy()

        for i, step in enumerate(workflow_config):
            step_name = step.get("name", f"step_{i}")
            operation = step["operation"]
            params = step.get("params", {})

            # Check cache
            cache_key = None
            if cache_intermediate:
                cache_key = self._get_cache_key(current_graph, operation, params)
                if cache_key in self._cache:
                    logger.info(f"Using cached result for {step_name}")
                    step_result = self._cache[cache_key]
                else:
                    step_result = self._execute_operation(
                        current_graph, operation, params
                    )
                    self._cache[cache_key] = step_result
            else:
                step_result = self._execute_operation(current_graph, operation, params)

            results[step_name] = step_result

            # Update graph if operation modifies it
            if step.get("modifies_graph", False):
                current_graph = step_result.get("graph", current_graph)

            # Early termination condition
            if step.get("stop_condition"):
                condition = step["stop_condition"]
                if self._evaluate_condition(step_result, condition):
                    logger.info(f"Workflow stopped at {step_name} due to condition")
                    break

        return {
            "workflow_results": results,
            "final_graph": current_graph,
            "steps_completed": len(results),
            "total_time": time.time() - start_time,
            "cache_hits": sum(1 for k in results if k in self._cache),
        }

    def report_generation(
        self,
        graph_analysis: dict[str, Any],
        template: str = "default",
        output_format: str = "pdf",
        include_visualizations: bool = True,
        **kwargs,
    ) -> bytes | str:
        """
        Generate automated PDF/HTML reports.

        Parameters:
        -----------
        graph_analysis : dict
            Analysis results to include in report
        template : str
            Report template to use
        format : str
            Output format: 'pdf' or 'html'
        include_visualizations : bool
            Include graph visualizations

        Returns:
        --------
        Report content (bytes for PDF, string for HTML)
        """
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")

        if format == "html":
            return self._generate_html_report(
                graph_analysis, template, timestamp, include_visualizations, **kwargs
            )
        elif format == "pdf":
            return self._generate_pdf_report(
                graph_analysis, template, timestamp, include_visualizations, **kwargs
            )
        else:
            msg = f"Unsupported format: {output_format}"
            raise ValueError(msg)

    def alert_system(
        self,
        graph: nx.Graph,
        alert_rules: list[dict[str, Any]],
        notification_callback: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """
        Anomaly detection and alerting system.

        Parameters:
        -----------
        graph : nx.Graph
            Graph to monitor
        alert_rules : list
            List of alert rule configurations
        notification_callback : callable
            Function to call for notifications

        Returns:
        --------
        List of triggered alerts
        """
        triggered_alerts = []

        for rule in alert_rules:
            rule_name = rule["name"]
            condition_type = rule["type"]

            if condition_type == "threshold":
                # Threshold-based alerts
                metric = rule["metric"]
                threshold = rule["threshold"]
                operator = rule.get("operator", "gt")

                # Calculate metric
                if metric == "density":
                    value = nx.density(graph)
                elif metric == "avg_degree":
                    value = sum(d for n, d in graph.degree()) / graph.number_of_nodes()
                elif metric == "num_components":
                    value = nx.number_connected_components(graph)
                elif metric == "diameter":
                    if nx.is_connected(graph):
                        value = nx.diameter(graph)
                    else:
                        value = float("inf")
                else:
                    continue

                # Check condition
                triggered = False
                if operator == "gt" and value > threshold:
                    triggered = True
                elif operator == "lt" and value < threshold:
                    triggered = True
                elif operator == "eq" and value == threshold:
                    triggered = True

                if triggered:
                    alert = {
                        "rule_name": rule_name,
                        "type": "threshold",
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "operator": operator,
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "severity": rule.get("severity", "medium"),
                    }
                    triggered_alerts.append(alert)

            elif condition_type == "anomaly":
                # Statistical anomaly detection
                metric = rule["metric"]
                sensitivity = rule.get("sensitivity", 2.0)  # Standard deviations

                # Get historical values (would come from time series in production)
                # For demo, generate synthetic historical data
                historical_values = self._get_historical_metrics(
                    graph, metric, periods=30
                )
                current_value = historical_values[-1]

                mean = np.mean(historical_values[:-1])
                std = np.std(historical_values[:-1])

                if abs(current_value - mean) > sensitivity * std:
                    alert = {
                        "rule_name": rule_name,
                        "type": "anomaly",
                        "metric": metric,
                        "value": current_value,
                        "mean": mean,
                        "std": std,
                        "z_score": (current_value - mean) / std if std > 0 else 0,
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "severity": rule.get("severity", "high"),
                    }
                    triggered_alerts.append(alert)

            elif condition_type == "pattern":
                # Pattern-based alerts (e.g., sudden component split)
                pattern = rule["pattern"]

                if pattern == "component_split":
                    # Check if graph recently split into more components
                    # (would track over time in production)
                    if not nx.is_connected(graph):
                        alert = {
                            "rule_name": rule_name,
                            "type": "pattern",
                            "pattern": pattern,
                            "num_components": nx.number_connected_components(graph),
                            "timestamp": datetime.now(tz=UTC).isoformat(),
                            "severity": rule.get("severity", "high"),
                        }
                        triggered_alerts.append(alert)

        # Send notifications
        if notification_callback and triggered_alerts:
            notification_callback(triggered_alerts)

        return triggered_alerts

    def schedule_job(
        self, job_config: dict[str, Any], start_immediately: bool = True
    ) -> str:
        """
        Schedule a job for execution.

        Parameters:
        -----------
        job_config : dict
            Job configuration including schedule and operation
        start_immediately : bool
            Start scheduler immediately

        Returns:
        --------
        str
            Job ID
        """
        if not HAS_SCHEDULE:
            raise ImportError(
                "schedule is required for job scheduling. Install with: pip install schedule"
            )
        # Using MD5 for non-cryptographic job ID generation (not security sensitive)
        job_id = hashlib.md5(
            json.dumps(job_config, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()[:8]

        job_config["job_id"] = job_id
        job_config["created_at"] = datetime.now(tz=UTC).isoformat()
        job_config["last_run"] = None
        job_config["next_run"] = None
        job_config["run_count"] = 0

        self._scheduled_jobs.append(job_config)

        # Configure schedule
        schedule_type = job_config["schedule"]["type"]

        if schedule_type == "interval":
            interval = job_config["schedule"]["interval"]
            unit = job_config["schedule"]["unit"]

            if unit == "seconds":
                schedule.every(interval).seconds.do(self._run_scheduled_job, job_id)
            elif unit == "minutes":
                schedule.every(interval).minutes.do(self._run_scheduled_job, job_id)
            elif unit == "hours":
                schedule.every(interval).hours.do(self._run_scheduled_job, job_id)
            elif unit == "days":
                schedule.every(interval).days.do(self._run_scheduled_job, job_id)

        elif schedule_type == "daily":
            time_str = job_config["schedule"]["time"]
            schedule.every().day.at(time_str).do(self._run_scheduled_job, job_id)

        elif schedule_type == "weekly":
            day = job_config["schedule"]["day"]
            time_str = job_config["schedule"]["time"]
            getattr(schedule.every(), day.lower()).at(time_str).do(
                self._run_scheduled_job, job_id
            )

        # Start scheduler thread if needed
        if start_immediately and self._scheduler_thread is None:
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop, daemon=True
            )
            self._scheduler_thread.start()

        logger.info(f"Scheduled job {job_id}: {job_config['name']}")

        return job_id

    def save_graph_version(
        self,
        graph: nx.Graph,
        version_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a versioned snapshot of a graph.

        Parameters:
        -----------
        graph : nx.Graph
            Graph to version
        version_name : str
            Version name/tag
        metadata : dict, optional
            Additional metadata

        Returns:
        --------
        Version ID
        """
        # Using MD5 for non-cryptographic version ID generation (not security sensitive)
        version_id = hashlib.md5(
            f"{version_name}_{datetime.now(tz=UTC).isoformat()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        version_data = {
            "version_id": version_id,
            "version_name": version_name,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "graph_info": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "type": type(graph).__name__,
            },
            "metadata": metadata or {},
        }

        # Save graph and metadata
        version_dir = self.cache_dir / "versions" / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save graph
        nx.write_gpickle(graph, version_dir / "graph.gpickle")

        # Save metadata
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(version_data, f, indent=2)

        # Update version index
        self._update_version_index(version_id, version_data)

        logger.info(f"Created version {version_id}: {version_name}")

        return version_id

    def _process_single_graph(
        self, graph_id: str, graph: nx.Graph, operations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Process a single graph with multiple operations."""
        results = {"graph_id": graph_id, "operations": {}}

        for op in operations:
            op_name = op["name"]
            op_type = op["type"]
            params = op.get("params", {})

            try:
                if op_type == "centrality":
                    result = self._calculate_centrality(graph, **params)
                elif op_type == "community":
                    result = self._detect_communities(graph, **params)
                elif op_type == "metrics":
                    result = self._calculate_metrics(graph, **params)
                else:
                    result = {"error": f"Unknown operation type: {op_type}"}

                results["operations"][op_name] = result

            except Exception as e:
                logger.error(f"Error in operation {op_name}: {e}")
                results["operations"][op_name] = {"error": str(e)}

        return results

    def _calculate_centrality(self, graph: nx.Graph, **params) -> dict[str, Any]:
        """Calculate centrality metrics."""
        centrality_type = params.get("type", "degree")

        if centrality_type == "degree":
            centrality = nx.degree_centrality(graph)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(graph)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(graph)
        elif centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(graph, max_iter=1000)
        else:
            centrality = {}

        return {
            "type": centrality_type,
            "values": centrality,
            "top_10": sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        }

    def _detect_communities(self, graph: nx.Graph, **params) -> dict[str, Any]:
        """Detect communities in graph."""
        # Simplified community detection
        if HAS_LOUVAIN:
            partition = community_louvain.best_partition(graph)

            communities = {}
            for node, comm in partition.items():
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)

            return {
                "num_communities": len(communities),
                "communities": communities,
                "modularity": community_louvain.modularity(partition, graph),
            }
        else:
            # Fallback to connected components
            if graph.is_directed():
                components = list(nx.weakly_connected_components(graph))
            else:
                components = list(nx.connected_components(graph))

            return {
                "num_communities": len(components),
                "communities": {i: list(comp) for i, comp in enumerate(components)},
                "method": "connected_components",
            }

    def _calculate_metrics(self, graph: nx.Graph, **params) -> dict[str, Any]:
        """Calculate graph metrics."""
        metrics = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": (
                nx.is_connected(graph)
                if not graph.is_directed()
                else nx.is_weakly_connected(graph)
            ),
        }

        if (
            metrics["is_connected"]
            and graph.number_of_nodes() < MAX_NODES_FOR_EXPENSIVE_METRICS
        ):
            metrics["diameter"] = nx.diameter(graph)
            metrics["radius"] = nx.radius(graph)

        if not graph.is_directed():
            metrics["average_clustering"] = nx.average_clustering(graph)
            metrics["transitivity"] = nx.transitivity(graph)

        return metrics

    def _get_cache_key(self, graph: nx.Graph, operation: str, params: dict) -> str:
        """Generate cache key for operation."""
        # Using MD5 for non-cryptographic cache key generation (not security sensitive)
        graph_hash = hashlib.md5(
            f"{graph.number_of_nodes()}_{graph.number_of_edges()}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:8]

        # Using MD5 for non-cryptographic params hash (not security sensitive)
        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()[:8]

        return f"{operation}_{graph_hash}_{params_hash}"

    def _execute_operation(
        self, graph: nx.Graph, operation: str, params: dict
    ) -> dict[str, Any]:
        """Execute a single operation on graph."""
        # Map operation names to functions
        operations = {
            "centrality": self._calculate_centrality,
            "community": self._detect_communities,
            "metrics": self._calculate_metrics,
            # Add more operations as needed
        }

        if operation in operations:
            return operations[operation](graph, **params)
        else:
            msg = f"Unknown operation: {operation}"
            raise ValueError(msg)

    def _evaluate_condition(
        self, result: dict[str, Any], condition: dict[str, Any]
    ) -> bool:
        """Evaluate workflow condition."""
        field = condition["field"]
        operator = condition["operator"]
        value = condition["value"]

        # Navigate nested fields
        current = result
        for part in field.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False

        # Evaluate condition
        if operator == "eq":
            return current == value
        elif operator == "ne":
            return current != value
        elif operator == "gt":
            return current > value
        elif operator == "lt":
            return current < value
        elif operator == "gte":
            return current >= value
        elif operator == "lte":
            return current <= value
        elif operator == "in":
            return current in value
        elif operator == "contains":
            return value in current
        else:
            return False

    def _generate_html_report(
        self,
        analysis: dict[str, Any],
        template: str,
        timestamp: str,
        include_viz: bool,
        **kwargs,
    ) -> str:
        """Generate HTML report."""
        if not HAS_JINJA2:
            raise ImportError(
                "jinja2 is required for HTML report generation. Install with: pip install jinja2"
            )
        # Simple HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Network Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { background-color: #e8f4f8; padding: 10px; margin: 10px 0; }
                .visualization { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Network Analysis Report</h1>
            <p>Generated: {{ timestamp }}</p>

            <h2>Summary</h2>
            <div class="metric">
                <p>Nodes: {{ num_nodes }}</p>
                <p>Edges: {{ num_edges }}</p>
                <p>Density: {{ density }}</p>
            </div>

            {% if visualizations %}
            <h2>Visualizations</h2>
            {% for viz in visualizations %}
            <div class="visualization">
                <h3>{{ viz.title }}</h3>
                <img src="{{ viz.data }}" style="max-width: 100%;">
            </div>
            {% endfor %}
            {% endif %}

            <h2>Detailed Results</h2>
            {{ detailed_results }}
        </body>
        </html>
        """

        template_obj = Template(html_template)

        # Prepare data
        context = {
            "timestamp": timestamp,
            "num_nodes": analysis.get("num_nodes", 0),
            "num_edges": analysis.get("num_edges", 0),
            "density": analysis.get("density", 0),
            "visualizations": [],
            "detailed_results": self._format_detailed_results(analysis),
        }

        if include_viz and "visualizations" in analysis:
            context["visualizations"] = analysis["visualizations"]

        return template_obj.render(**context)

    def _generate_pdf_report(
        self,
        analysis: dict[str, Any],
        template: str,
        timestamp: str,
        include_viz: bool,
        **kwargs,
    ) -> bytes:
        """Generate PDF report."""
        if not HAS_REPORTLAB:
            raise ImportError(
                "reportlab is required for PDF generation. Install with: pip install reportlab"
            )
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor="HexColor(0x333333)",
        )
        story.append(Paragraph("Network Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Timestamp
        story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Summary section
        story.append(Paragraph("Summary", styles["Heading2"]))
        summary_data = [
            ["Metric", "Value"],
            ["Number of Nodes", str(analysis.get("num_nodes", 0))],
            ["Number of Edges", str(analysis.get("num_edges", 0))],
            ["Density", f"{analysis.get('density', 0):.4f}"],
            ["Connected", str(analysis.get("is_connected", False))],
        ]

        summary_table = Table(summary_data)
        summary_table.setStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), "HexColor(0xCCCCCC)"),
                ("GRID", (0, 0), (-1, -1), 1, "HexColor(0x000000)"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Visualizations
        if include_viz and "visualizations" in analysis:
            story.append(Paragraph("Visualizations", styles["Heading2"]))
            for viz in analysis["visualizations"]:
                if viz.get("type") == "image" and viz.get("data"):
                    # Decode base64 image
                    img_data = base64.b64decode(viz["data"].split(",")[1])
                    img = Image(io.BytesIO(img_data), width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 12))

        # Detailed results
        story.append(PageBreak())
        story.append(Paragraph("Detailed Results", styles["Heading2"]))
        story.append(
            Paragraph(self._format_detailed_results(analysis), styles["Normal"])
        )

        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    def _format_detailed_results(self, analysis: dict[str, Any]) -> str:
        """Format analysis results for report."""
        # Convert nested dict to readable format
        lines = []

        def format_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append("  " * indent + f"{key}:")
                    format_dict(value, indent + 1)
                elif (
                    isinstance(value, list)
                    and len(value) > 0
                    and isinstance(value[0], (list, tuple))
                ):
                    lines.append("  " * indent + f"{key}:")
                    for item in value[:DISPLAY_LIMIT_FOR_ITEMS]:  # Limit to first items
                        lines.append("  " * (indent + 1) + str(item))
                    if len(value) > DISPLAY_LIMIT_FOR_ITEMS:
                        lines.append(
                            "  " * (indent + 1)
                            + f"... and {len(value) - DISPLAY_LIMIT_FOR_ITEMS} more"
                        )
                else:
                    lines.append("  " * indent + f"{key}: {value}")

        format_dict(analysis)
        return "\n".join(lines)

    def _get_historical_metrics(
        self, graph: nx.Graph, metric: str, periods: int = 30
    ) -> list[float]:
        """Get historical metric values (simulated for demo)."""
        # In production, this would query a time series database
        current_value = 0

        if metric == "density":
            current_value = nx.density(graph)
        elif metric == "avg_degree":
            current_value = sum(d for n, d in graph.degree()) / graph.number_of_nodes()
        elif metric == "clustering":
            current_value = (
                nx.average_clustering(graph) if not graph.is_directed() else 0
            )

        # Generate synthetic historical data with some variation
        np.random.seed(42)
        historical = []
        base_value = current_value * 0.9

        for _i in range(periods - 1):
            noise = np.random.normal(0, current_value * 0.1)
            historical.append(max(0, base_value + noise))

        historical.append(current_value)
        return historical

    def _run_scheduled_job(self, job_id: str):
        """Execute a scheduled job."""
        job = next((j for j in self._scheduled_jobs if j["job_id"] == job_id), None)

        if not job:
            logger.error(f"Job {job_id} not found")
            return

        logger.info(f"Running scheduled job {job_id}: {job['name']}")

        try:
            # Execute job operation
            operation = job["operation"]
            result = self._execute_operation(**operation)

            # Update job metadata
            job["last_run"] = datetime.now(tz=UTC).isoformat()
            job["run_count"] += 1
            job["last_result"] = result

            # Store results if configured
            if job.get("store_results"):
                results_dir = self.cache_dir / "job_results" / job_id
                results_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
                with open(results_dir / f"result_{timestamp}.json", "w") as f:
                    json.dump(result, f, indent=2)

            # Trigger notifications if configured
            if job.get("notifications") and result.get("alerts"):
                self._send_job_notifications(job, result["alerts"])

        except Exception as e:
            logger.error(f"Error in scheduled job {job_id}: {e}")
            job["last_error"] = str(e)

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while True:
            schedule.run_pending()
            time.sleep(1)

    def _update_version_index(self, version_id: str, version_data: dict[str, Any]):
        """Update version control index."""
        index_file = self.cache_dir / "versions" / "index.json"

        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
        else:
            index = {"versions": []}

        index["versions"].append(
            {
                "version_id": version_id,
                "version_name": version_data["version_name"],
                "timestamp": version_data["timestamp"],
                "graph_info": version_data["graph_info"],
            }
        )

        # Keep only last 100 versions in index
        index["versions"] = index["versions"][-100:]

        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

    def _send_job_notifications(
        self, job: dict[str, Any], alerts: list[dict[str, Any]]
    ):
        """Send notifications for scheduled job alerts."""
        # In production, integrate with notification services
        logger.info(f"Sending {len(alerts)} alerts for job {job['job_id']}")
        for alert in alerts:
            logger.warning(f"Alert: {alert}")
