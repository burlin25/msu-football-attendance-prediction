"""Render the attendance history and chronological evaluation figures."""

import json
import math
from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .model import MODEL_LABELS

INK = "#19352e"
MUTED = "#526660"
GRID = "#dde8e3"
HOLDOUT_SHADE = "#e5f0eb"
MODEL_COLORS = {
    "season_median": "#627674",
    "ridge": "#187850",
    "random_forest": "#a95820",
}
MODEL_MARKERS = {"season_median": "x", "ridge": "s", "random_forest": "^"}


def _figure(title: str, subtitle: str, size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, facecolor="white")
    figure.text(0.10, 0.95, title, color=INK, size=20, weight="bold", va="top")
    figure.text(0.10, 0.89, subtitle, color=MUTED, size=10, va="top")
    return figure


def _axis_style(axis, grid_axis: str = "y") -> None:
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axis.tick_params(length=0, colors=MUTED, labelsize=9, pad=7)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _thousands(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if value else "0"


def _attendance_history(
    games: pd.DataFrame, holdout_seasons: list[int], destination: Path
) -> None:
    figure = _figure(
        "Reported attendance at Spartan Stadium",
        "Each point is one game. The line connects season medians.",
        (10.8, 5.1),
    )
    axis = figure.subplots()
    figure.subplots_adjust(left=0.10, right=0.96, top=0.79, bottom=0.21)
    season = games["season"]
    year_range = range(int(season.min()), int(season.max()) + 1)
    medians = games.groupby("season")["attendance"].median().reindex(year_range)
    game_offsets = (
        games.groupby("season").cumcount()
        - (games.groupby("season")["game_date"].transform("size") - 1) / 2
    )
    axis.axvspan(
        holdout_seasons[0] - 0.5, holdout_seasons[-1] + 0.5, color=HOLDOUT_SHADE
    )
    axis.axvspan(2019.5, 2020.5, color="#f1f2ef")
    axis.scatter(
        season + 0.06 * game_offsets,
        games["attendance"],
        color="#619682",
        alpha=0.78,
        s=19,
    )
    axis.plot(
        medians.index, medians, color=INK, linewidth=2.1, marker="o", markersize=3.4
    )
    axis.set_xlim(year_range.start - 0.5, year_range.stop - 0.5)
    ticks = sorted(set(year_range[::2]) | {year_range[-1]})
    if len(ticks) >= 2 and ticks[-1] - ticks[-2] == 1:
        del ticks[-2]
    axis.set_xticks(ticks)
    axis.set_ylabel("Reported attendance", color=MUTED, size=10)
    axis.yaxis.set_major_formatter(FuncFormatter(_thousands))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    _axis_style(axis)
    figure.text(
        0.10,
        0.075,
        "Gray band marks 2020, when public admission was restricted. Green band marks the held-out seasons.",
        size=9,
        color=MUTED,
    )
    figure.text(
        0.10, 0.038, "Source data from Michigan State Athletics.", size=8, color=MUTED
    )
    figure.savefig(destination, dpi=180)


def _holdout_predictions(
    predictions: pd.DataFrame, metrics: dict, destination: Path
) -> None:
    holdout_seasons = metrics["holdout_seasons"]
    last_train_season = metrics["holdout_baseline_season"]
    figure = _figure(
        "Attendance predictions for the held-out games",
        f"Models trained through {last_train_season}. The baseline is the {last_train_season} median.",
        (10.8, 7.4),
    )
    axes = figure.subplots(len(holdout_seasons), sharey=True)
    figure.subplots_adjust(left=0.10, right=0.96, top=0.77, bottom=0.14, hspace=0.63)
    for axis, season in zip(axes, holdout_seasons, strict=True):
        rows = predictions[predictions["season"] == season]
        games = rows.pivot(
            index=["game_date", "opponent", "attendance"],
            columns="model",
            values="prediction",
        )
        games = games.reset_index().sort_values("game_date")
        x = np.arange(len(games))
        axis.plot(
            x,
            games["attendance"],
            label="Reported attendance",
            color=INK,
            linewidth=2.2,
            marker="o",
            zorder=4,
        )
        for name, label in MODEL_LABELS.items():
            axis.plot(
                x,
                games[name],
                label=label,
                color=MODEL_COLORS[name],
                linewidth=1.4,
                linestyle="--" if name == "season_median" else ":",
                marker=MODEL_MARKERS[name],
                markersize=5.2,
            )
        labels = [
            f"{date:%b %d}\n{fill(opponent, 14)}"
            for date, opponent in zip(
                games["game_date"], games["opponent"], strict=True
            )
        ]
        axis.set_xticks(x, labels)
        axis.set_title(
            f"{season}   ·   {len(games)} games",
            loc="left",
            color=INK,
            size=11,
            weight="bold",
            pad=7,
        )
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_formatter(FuncFormatter(_thousands))
        _axis_style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncols=4,
        loc="center left",
        bbox_to_anchor=(0.093, 0.825),
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.10,
        0.037,
        "Games are evenly spaced in each panel. Both panels use the same attendance scale.",
        size=9,
        color=MUTED,
    )
    figure.savefig(destination, dpi=180)


def _model_errors(metrics: dict, destination: Path) -> None:
    holdout = metrics["holdout"]
    weather = metrics["weather_sensitivity"]
    first, last = metrics["holdout_seasons"]
    figure = _figure(
        "How far off were the predictions?",
        f"Mean absolute error in attendees on the {first}–{last} holdout. Lower is better.",
        (10.8, 4.9),
    )
    primary, sensitivity = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.17, right=0.95, top=0.72, bottom=0.20, wspace=0.47)
    errors = [entry["mae"] for entry in holdout.values()]
    limit = math.ceil(max(errors) * 1.22 / 1000) * 1000 or 1000
    rows = list(MODEL_LABELS)
    primary.barh(
        range(len(rows)),
        [holdout[name]["mae"] for name in rows],
        color=[MODEL_COLORS[name] for name in rows],
        height=0.48,
    )
    primary.set_yticks(range(len(rows)), [MODEL_LABELS[name] for name in rows])
    primary.invert_yaxis()
    primary.set_title(
        f"Pregame models   ·   {metrics['n_holdout']} games",
        loc="left",
        color=INK,
        size=11,
        weight="bold",
        pad=9,
    )
    for index, name in enumerate(rows):
        mae = holdout[name]["mae"]
        primary.text(
            mae + 0.025 * limit, index, f"{mae:,.0f}", va="center", color=INK, size=9
        )
    primary.set_xlim(0, limit)
    primary.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    primary.xaxis.set_major_formatter(FuncFormatter(_thousands))
    _axis_style(primary, "x")

    if weather is None:
        sensitivity.set_axis_off()
        sensitivity.text(
            0.06,
            0.58,
            "Insufficient matched weather observations",
            color=MUTED,
            size=10,
        )
    else:
        changes = weather["mae_change"]
        weather_limit = max(4, max(abs(change) for change in changes.values()) * 1.48)
        for index, (name, change) in enumerate(changes.items()):
            sensitivity.barh(index, change, color=MODEL_COLORS[name], height=0.38)
            label = f"{change:+,.0f}" if abs(change) >= 0.5 else "0"
            sensitivity.text(
                change + math.copysign(0.045 * weather_limit, change),
                index,
                label,
                ha="left" if change >= 0 else "right",
                va="center",
                color=INK,
                size=9,
            )
        sensitivity.set_yticks(
            range(len(changes)), [MODEL_LABELS[name] for name in changes]
        )
        sensitivity.set_ylim(len(changes) - 0.5, -0.5)
        sensitivity.set_xlim(-weather_limit, weather_limit)
        sensitivity.axvline(0, color=MUTED, linewidth=0.9)
        sensitivity.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        sensitivity.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:+,.0f}" if value else "0")
        )
        _axis_style(sensitivity, "x")
        sensitivity.set_title(
            f"Change with weather   ·   {weather['n_holdout']} games",
            loc="left",
            color=INK,
            size=11,
            weight="bold",
            pad=9,
        )
        figure.text(
            0.63,
            0.095,
            "Error change after adding observed weather",
            size=9,
            color=MUTED,
        )
        figure.text(
            0.10,
            0.037,
            "The weather comparison uses the same training and test games for both versions.",
            size=8,
            color=MUTED,
        )
    figure.savefig(destination, dpi=180)


def render(games: pd.DataFrame, output_dir: Path) -> None:
    """Render figures from the clean games and the reports written by evaluate."""
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "predictions.csv", parse_dates=["game_date"])
    holdout = predictions[
        (predictions["split"] == "holdout") & (predictions["experiment"] == "pregame")
    ]
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _attendance_history(
        games.sort_values("game_date"),
        metrics["holdout_seasons"],
        figures / "attendance-history.png",
    )
    _holdout_predictions(holdout, metrics, figures / "holdout-predictions.png")
    _model_errors(metrics, figures / "model-errors.png")
