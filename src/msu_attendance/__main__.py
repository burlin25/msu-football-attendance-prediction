"""Run the attendance project from the repository root."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("fetch", "download and cache public source files"),
        ("run", "fetch, build and evaluate in one command"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--refresh", action="store_true", help="download new copies of cached inputs")
    commands.add_parser("build", help="validate and join previously cached data")
    commands.add_parser("evaluate", help="regenerate metrics and figures from the processed CSV")
    args = parser.parse_args()

    if args.command in {"fetch", "build", "run"}:
        from .pipeline import build, download

        if args.command in {"fetch", "run"}:
            download(refresh=args.refresh)
        if args.command in {"build", "run"}:
            games = build()
            print(f"Built {len(games)} games, including {games['model_eligible'].sum()} eligible for modeling")

    if args.command in {"evaluate", "run"}:
        import pandas as pd

        from .model import evaluate
        from .plots import render

        games = pd.read_csv("data/processed/games.csv")
        model_games = games.loc[games["model_eligible"]].copy()
        evaluate(model_games, Path("reports"))
        render(model_games, Path("reports"))
        print("Wrote model results and figures to reports/")


if __name__ == "__main__":
    main()
