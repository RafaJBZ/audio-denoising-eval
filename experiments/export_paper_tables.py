from __future__ import annotations

from pathlib import Path

from _bootstrap import ensure_project_root_on_path


def main() -> None:
    ensure_project_root_on_path()
    from src.eval.export_paper import (
        export_latency_table,
        export_metrics_table,
        export_quality_latency_tradeoff_figure,
        export_training_cost_table,
    )

    export_metrics_table(
        metrics_json_path=Path("experiments/metrics_report.json"),
        output_table_path=Path("paper/tables/metrics_table.tex"),
    )
    export_latency_table(
        latency_json_path=Path("experiments/latency_report.json"),
        output_table_path=Path("paper/tables/latency_table.tex"),
    )
    export_training_cost_table(
        mlruns_root=Path("experiments/mlruns"),
        output_table_path=Path("paper/tables/training_cost_table.tex"),
    )
    export_quality_latency_tradeoff_figure(
        metrics_json_path=Path("experiments/metrics_report.json"),
        latency_json_path=Path("experiments/latency_report.json"),
        output_figure_path=Path("paper/figures/quality_latency_tradeoff.png"),
    )
    print("Exported paper tables and quality-latency tradeoff figure.")


if __name__ == "__main__":
    main()

