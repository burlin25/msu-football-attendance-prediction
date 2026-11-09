"""Join NOAA airport observations to Spartan Stadium kickoffs."""

from datetime import timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from .sources import fetch

STATION = "USW00014836"
# NOAA LCDv2 dates are local standard time year-round, including during EDT.
STATION_STANDARD_TIME = timezone(timedelta(hours=-5))
MAX_OBSERVATION_AGE = pd.Timedelta(minutes=75)
NOAA_FIELDS = (
    "HourlyDryBulbTemperature",
    "HourlyPrecipitation",
    "HourlyWindSpeed",
)
WEATHER_COLUMNS = [
    "weather_observed_at",
    "weather_station",
    "weather_source_url",
    "temperature_f",
    "precipitation_in",
    "weather_precipitation_trace",
    "wind_speed_mph",
]


def weather_url(year: int) -> str:
    query = urlencode(
        {
            "dataset": "local-climatological-data-v2",
            "stations": STATION,
            "startDate": f"{year}-08-01",
            "endDate": f"{year}-12-31",
            "dataTypes": ",".join(NOAA_FIELDS),
            "format": "csv",
            "units": "metric",
        }
    )
    return f"https://www.ncei.noaa.gov/access/services/data/v1?{query}"


def load_weather(year: int, raw_dir: Path, *, refresh: bool = False) -> pd.DataFrame:
    """Load routine and special METAR observations for the football season."""
    url = weather_url(year)
    path = fetch(url, raw_dir / "weather" / f"noaa_lcdv2_{year}.csv", refresh=refresh)
    raw = pd.read_csv(path, usecols=["STATION", "DATE", "REPORT_TYPE", *NOAA_FIELDS], dtype="string")
    if raw.empty or not raw["STATION"].fillna("").eq(STATION).all():
        raise ValueError(f"Expected NOAA observations for {STATION} in {path}")

    raw = raw.loc[raw["REPORT_TYPE"].isin(["FM-15", "FM-16"])]
    observed_at = pd.to_datetime(raw["DATE"], format="%Y-%m-%dT%H:%M:%S")
    observed_at = observed_at.dt.tz_localize(STATION_STANDARD_TIME).dt.tz_convert("UTC")
    temperature_c = pd.to_numeric(raw["HourlyDryBulbTemperature"], errors="coerce")
    wind_speed_ms = pd.to_numeric(raw["HourlyWindSpeed"], errors="coerce")
    raw_precipitation = raw["HourlyPrecipitation"].str.strip()
    precipitation_mm = pd.to_numeric(raw_precipitation, errors="coerce")
    # Trace means observed but below measurable precision. Preserve it separately.
    trace = raw_precipitation.eq("T")
    trace = trace.where(precipitation_mm.notna() | trace.fillna(False))

    weather = pd.DataFrame(
        {
            "weather_observed_at": observed_at,
            "weather_station": STATION,
            "weather_source_url": url,
            "temperature_f": (temperature_c * 9 / 5 + 32).round(2),
            "precipitation_in": (precipitation_mm / 25.4).round(3),
            "weather_precipitation_trace": trace,
            "wind_speed_mph": (wind_speed_ms / 0.44704).round(2),
        }
    ).dropna(subset=["weather_observed_at", "temperature_f"])
    if weather.empty:
        raise ValueError(f"No usable hourly NOAA observations in {path}")

    # At an identical timestamp, prefer the report with more measured fields.
    weather["_completeness"] = weather[["weather_precipitation_trace", "wind_speed_mph"]].notna().sum(axis=1)
    return (
        weather.sort_values("_completeness", ascending=False, kind="stable")
        .drop_duplicates("weather_observed_at")
        .sort_values("weather_observed_at")
        .loc[:, WEATHER_COLUMNS]
        .reset_index(drop=True)
    )


def attach_weather(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Use the last observation at or before kickoff and preserve unmatched games."""
    kickoffs = games["kickoff_local"]
    if "is_spartan_stadium" in games:
        kickoffs = kickoffs.where(games["is_spartan_stadium"])
    kickoffs = kickoffs.reset_index(drop=True)
    if kickoffs.notna().any():
        kickoffs = kickoffs.dt.tz_convert("UTC")
    else:
        kickoffs = pd.to_datetime(kickoffs, utc=True)
    kickoffs = kickoffs.astype("datetime64[ns, UTC]")
    lookup = (
        kickoffs.dropna()
        .rename("kickoff_utc")
        .rename_axis("game_row")
        .reset_index()
        .sort_values("kickoff_utc")
    )
    matches = pd.merge_asof(
        lookup,
        weather.sort_values("weather_observed_at"),
        left_on="kickoff_utc",
        right_on="weather_observed_at",
        tolerance=MAX_OBSERVATION_AGE,
        direction="backward",
    )
    matches = matches.set_index("game_row")[WEATHER_COLUMNS].reindex(range(len(games)))
    return pd.concat([games, matches.set_axis(games.index)], axis=1)
