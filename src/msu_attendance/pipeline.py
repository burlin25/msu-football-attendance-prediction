"""Validate source coverage and create a documented, game-level dataset."""

import json
from pathlib import Path

import pandas as pd

from .games import load_games, schedule_url, stats_url
from .sources import describe, fetch
from .weather import attach_weather, load_weather, weather_url

SEASONS = range(2005, 2026)
RESTRICTED_SEASON = 2020
PROCESSED_GAMES = Path("data/processed/games.csv")
RAW_DIR = Path("data/raw")
REPORTS = Path("reports")


def _season_sources(season: int) -> tuple[tuple[str, Path], ...]:
    return (
        (schedule_url(season), RAW_DIR / "msu" / f"schedule-{season}.html"),
        (stats_url(season), RAW_DIR / "msu" / f"stats-{season}.html"),
        (weather_url(season), RAW_DIR / "weather" / f"noaa_lcdv2_{season}.csv"),
    )


def download(*, refresh: bool = False) -> None:
    for season in SEASONS:
        for url, path in _season_sources(season):
            fetch(url, path, refresh=refresh)
        print(f"Cached sources for {season}", flush=True)


def add_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values(["season", "game_date"]).reset_index(drop=True).copy()
    if games.duplicated(["season", "game_date"]).any():
        raise ValueError("A season/date appears more than once in the Michigan State schedule")
    if not games["result"].isin(["W", "L"]).all():
        raise ValueError("Expected an on-field win or loss for each completed game")

    # Subtract today's indicator from the running total to obtain the pregame record.
    for result, name in [("W", "prior_wins"), ("L", "prior_losses")]:
        indicator = games["result"].eq(result).astype(int)
        games[name] = indicator.groupby(games["season"]).cumsum() - indicator
    return games


def mark_model_scope(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["model_exclusion"] = ""
    games.loc[games["attendance"].isna() | games["attendance"].le(0), "model_exclusion"] = (
        "missing_or_nonpositive_attendance"
    )
    games.loc[games["season"].eq(RESTRICTED_SEASON), "model_exclusion"] = "restricted_2020_season"
    games.loc[~games["is_spartan_stadium"], "model_exclusion"] = "other_venue"
    games["model_eligible"] = games["model_exclusion"].eq("")
    return games


def _coverage(games: pd.DataFrame) -> dict:
    model = games.loc[games["model_eligible"]]
    stadium = games.loc[games["is_spartan_stadium"]]
    by_season = []
    for season, year in games.groupby("season"):
        home = year.loc[year["is_spartan_stadium"]]
        eligible = home.loc[home["model_eligible"]]
        by_season.append({
            "season": int(season),
            "all_games": len(year),
            "spartan_stadium_games": len(home),
            "stadium_attendance_reported": int(home["attendance"].count()),
            "model_games": len(eligible),
            "model_kickoff_known": int(eligible["kickoff_local"].count()),
            "model_weather_matched": int(eligible["weather_observed_at"].count()),
            "model_temperature_and_wind": int(eligible[["temperature_f", "wind_speed_mph"]].notna().all(axis=1).sum()),
            "model_precipitation_observed": int(eligible["precipitation_in"].count()),
            "model_precipitation_trace": int(eligible["weather_precipitation_trace"].sum()),
        })
    return {
        "all_games": len(games),
        "spartan_stadium_games": len(stadium),
        "spartan_stadium_attendance_reported": int(stadium["attendance"].count()),
        "model_games": len(model),
        "model_attendance_min": int(model["attendance"].min()),
        "model_attendance_max": int(model["attendance"].max()),
        "model_kickoff_known": int(model["kickoff_local"].count()),
        "model_weather_matched": int(model["weather_observed_at"].count()),
        "model_temperature_and_wind": int(model[["temperature_f", "wind_speed_mph"]].notna().all(axis=1).sum()),
        "model_precipitation_observed": int(model["precipitation_in"].count()),
        "model_precipitation_trace": int(model["weather_precipitation_trace"].sum()),
        "exclusions": games.loc[~games["model_eligible"], "model_exclusion"].value_counts().to_dict(),
        "by_season": by_season,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build() -> pd.DataFrame:
    source_records = []
    game_frames = []
    weather_frames = []
    for season in SEASONS:
        sources = _season_sources(season)
        for url, path in sources:
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Missing {path}. Run `python -m msu_attendance fetch` first.")
            source_records.append(describe(url, path))
        game_frames.append(load_games(season, sources[0][1], sources[1][1]))
        weather_frames.append(load_weather(season, RAW_DIR))
    games = add_pregame_features(pd.concat(game_frames, ignore_index=True))
    games = mark_model_scope(attach_weather(games, pd.concat(weather_frames, ignore_index=True)))
    audit = _coverage(games)
    PROCESSED_GAMES.parent.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    games.to_csv(PROCESSED_GAMES, index=False)
    _write_json(REPORTS / "data_audit.json", audit)
    _write_json(REPORTS / "sources.json", {"files": source_records})
    return games
