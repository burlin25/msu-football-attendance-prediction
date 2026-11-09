"""Evaluate pregame attendance models without training on future seasons."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HOLDOUT_SEASONS = (2024, 2025)
MIN_TRAIN_SEASONS = 5
PREGAME_NUMERIC = (
    "month",
    "is_saturday",
    "kickoff_hour",
    "is_conference",
    "prior_wins",
    "prior_losses",
)
WEATHER_NUMERIC = ("temperature_f", "wind_speed_mph")
MODEL_LABELS = {
    "season_median": "Prior-season median",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
}
RIDGE_PARAMS = {"alpha": 8.0}
FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_leaf": 3,
    "max_features": 0.75,
    "random_state": 42,
    "n_jobs": 1,
}
GAME_FIELDS = ["game_date", "season", "opponent", "attendance"]


def _preprocessor(numeric_features: tuple[str, ...]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                ),
                list(numeric_features),
            ),
            (
                "opponent",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["opponent"],
            ),
        ]
    )


def _predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    split: str,
    experiment: str = "pregame",
) -> pd.DataFrame:
    numeric_features = PREGAME_NUMERIC
    if experiment == "paired_weather":
        numeric_features += WEATHER_NUMERIC
    features = [*numeric_features, "opponent"]
    models = {
        "ridge": Ridge(**RIDGE_PARAMS),
        "random_forest": RandomForestRegressor(**FOREST_PARAMS),
    }
    predictions = {}
    if experiment == "pregame":
        latest_train_season = train["season"].max()
        season_median = train.loc[
            train["season"] == latest_train_season, "attendance"
        ].median()
        predictions["season_median"] = np.repeat(season_median, len(test))

    for name, model in models.items():
        pipeline = make_pipeline(_preprocessor(numeric_features), model)
        pipeline.fit(train[features], train["attendance"])
        predictions[name] = pipeline.predict(test[features])

    return pd.concat(
        [
            test[GAME_FIELDS].assign(
                split=split,
                experiment=experiment,
                model=name,
                prediction=values,
                absolute_error=np.abs(test["attendance"].to_numpy() - values),
            )
            for name, values in predictions.items()
        ],
        ignore_index=True,
    )


def _metrics(predictions: pd.DataFrame) -> dict:
    return {
        name: {
            "n": len(rows),
            "mae": float(mean_absolute_error(rows["attendance"], rows["prediction"])),
            "rmse": float(
                np.sqrt(mean_squared_error(rows["attendance"], rows["prediction"]))
            ),
        }
        for name, rows in predictions.groupby("model", sort=False)
    }


def _by_season(predictions: pd.DataFrame) -> dict:
    return {
        str(season): _metrics(rows) for season, rows in predictions.groupby("season")
    }


def evaluate(games: pd.DataFrame, output_dir: Path) -> dict:
    """Write game-level predictions and metrics from chronological evaluations."""
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values("game_date")
    games["month"] = games["game_date"].dt.month
    games["is_saturday"] = games["game_date"].dt.dayofweek.eq(5)

    seasons = sorted(int(season) for season in games["season"].unique())
    if tuple(seasons[-2:]) != HOLDOUT_SEASONS:
        raise ValueError(
            f"Expected the final two seasons to be {HOLDOUT_SEASONS}, got {seasons[-2:]}"
        )
    train_seasons = seasons[:-2]
    if len(train_seasons) <= MIN_TRAIN_SEASONS:
        raise ValueError(
            f"Need more than {MIN_TRAIN_SEASONS} training seasons for annual backtests"
        )

    train = games[games["season"] < HOLDOUT_SEASONS[0]]
    test = games[games["season"].isin(HOLDOUT_SEASONS)]
    holdout = _predict(train, test, "holdout")

    validation_seasons = train_seasons[MIN_TRAIN_SEASONS:]
    validation = pd.concat(
        [
            _predict(
                train[train["season"] < season],
                train[train["season"] == season],
                "validation",
            )
            for season in validation_seasons
        ],
        ignore_index=True,
    )

    metrics = {
        "training_seasons": train_seasons,
        "validation_seasons": validation_seasons,
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "n_train": len(train),
        "n_holdout": len(test),
        "holdout_baseline_season": train_seasons[-1],
        "model_settings": {"ridge": RIDGE_PARAMS, "random_forest": FOREST_PARAMS},
        "holdout": _metrics(holdout),
        "holdout_by_season": _by_season(holdout),
        "validation": _metrics(validation),
        "validation_by_season": _by_season(validation),
        "weather_sensitivity": None,
    }
    prediction_sets = [validation, holdout]
    weather_train = train.dropna(subset=list(WEATHER_NUMERIC))
    weather_test = test.dropna(subset=list(WEATHER_NUMERIC))
    if not weather_train.empty and not weather_test.empty:
        matched_pregame = _predict(
            weather_train, weather_test, "holdout", "paired_pregame"
        )
        matched_weather = _predict(
            weather_train, weather_test, "holdout", "paired_weather"
        )
        prediction_sets.extend([matched_pregame, matched_weather])
        no_weather_metrics = _metrics(matched_pregame)
        weather_metrics = _metrics(matched_weather)
        metrics["weather_sensitivity"] = {
            "n_train": len(weather_train),
            "n_holdout": len(weather_test),
            "pregame": no_weather_metrics,
            "with_weather": weather_metrics,
            "mae_change": {
                name: weather_metrics[name]["mae"] - no_weather_metrics[name]["mae"]
                for name in weather_metrics
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_sets, ignore_index=True)
    predictions.to_csv(
        output_dir / "predictions.csv", index=False, date_format="%Y-%m-%d"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics
