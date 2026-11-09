"""Regression checks for source reconciliation and pregame feature timing."""

import unittest
from pathlib import Path

import pandas as pd


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data" / "processed" / "games.csv"
        cls.games = pd.read_csv(path)
        cls.by_date = cls.games.set_index("game_date")

    def test_published_games_from_both_source_formats(self):
        for date, attendance, score, allowed in [
            ("2005-09-03", 73_949, (49, 14), True),  # Kent State, legacy report
            ("2010-09-18", 78_411, (34, 31), True),  # Notre Dame, venue verified by box score
            ("2025-08-29", 71_657, (23, 6), True),   # Western Michigan, modern report
            ("2025-11-29", 30_317, (38, 28), False),  # Maryland, Ford Field
        ]:
            with self.subTest(date=date):
                game = self.by_date.loc[date]
                self.assertEqual(game["attendance"], attendance)
                self.assertEqual((game["msu_score"], game["opponent_score"]), score)
                self.assertEqual(game["model_eligible"], allowed)

    def test_2020_rows_retained_outside_model(self):
        game = self.by_date.loc["2020-10-24"]
        self.assertEqual(game["attendance"], 376)
        self.assertEqual(game["model_exclusion"], "restricted_2020_season")
        self.assertFalse(self.games.loc[self.games["season"].eq(2020), "model_eligible"].any())

    def test_pregame_record_uses_earlier_games_at_all_venues(self):
        for _, season in self.games.sort_values("game_date").groupby("season"):
            for result, column in [("W", "prior_wins"), ("L", "prior_losses")]:
                previous = season["result"].eq(result).cumsum().shift(fill_value=0)
                self.assertEqual(season[column].tolist(), previous.tolist())

    def test_weather_join_is_local_and_never_uses_a_later_reading(self):
        with_weather = self.games.loc[self.games["weather_observed_at"].notna()]
        self.assertGreater(len(with_weather), 0)
        self.assertTrue(with_weather["is_spartan_stadium"].all())
        delay = (
            pd.to_datetime(with_weather["kickoff_local"], utc=True, format="mixed")
            - pd.to_datetime(with_weather["weather_observed_at"], utc=True, format="mixed")
        )
        self.assertTrue(delay.between(pd.Timedelta(0), pd.Timedelta(minutes=75)).all())
        western = self.by_date.loc["2025-08-29"]
        observed_local = pd.Timestamp(western["weather_observed_at"]).tz_convert("America/Detroit")
        self.assertEqual(observed_local.isoformat(), "2025-08-29T18:53:00-04:00")
        self.assertAlmostEqual(western["temperature_f"], 62.06)


if __name__ == "__main__":
    unittest.main()
