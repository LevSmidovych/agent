from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_chart_canvas_with_initial_figure(qapp) -> None:
    from benchmarks.charts import make_bar_figure
    from ui.chart_widget import ChartCanvas

    fig = make_bar_figure({"A": 10, "B": 20}, title="t", ylabel="y")
    canvas = ChartCanvas(figure=fig)
    assert canvas.figure is fig


def test_chart_canvas_set_figure_swaps_display(qapp) -> None:
    from benchmarks.charts import make_bar_figure, make_radar_figure
    from ui.chart_widget import ChartCanvas

    canvas = ChartCanvas(figure=make_bar_figure({"A": 1}, title="a", ylabel="b"))
    new_fig = make_radar_figure({"A": {"speed": 50}})
    canvas.set_figure(new_fig)
    assert canvas.figure is new_fig


def test_chart_canvas_save_png(qapp, tmp_path: Path) -> None:
    from benchmarks.charts import make_bar_figure
    from ui.chart_widget import ChartCanvas

    canvas = ChartCanvas(figure=make_bar_figure({"A": 10}, title="t", ylabel="y"))
    path = tmp_path / "out.png"
    canvas.save_png(path)
    assert path.exists()
    assert path.stat().st_size > 100


def test_chart_canvas_default_figure_when_none_provided(qapp) -> None:
    from ui.chart_widget import ChartCanvas

    canvas = ChartCanvas()
    assert canvas.figure is not None
