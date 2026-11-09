"""Join Michigan State Athletics schedules and reported game attendance."""

import json
import re
from datetime import datetime, time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from lxml import html

MSU_SITE = "https://msuspartans.com"
LEGACY_STATS = (
    "https://michstate_ftp.sidearmsports.com/custompages/sports/m-footbl/stats"
)
EASTERN = "America/Detroit"
NOTRE_DAME_2010_BOX_SCORE = (
    f"{MSU_SITE}/sports/2018/7/25/_m_footbl_stats_2010_2011_nd_msu_html"
)

LEGACY_HEADER = re.compile(r"\bDATE\s+OPPONENT\s+W/L\s+SCORE\s+ATTEND\b")
LEGACY_GAME = re.compile(
    r"^\s*\*?(?P<month>[A-Z][a-z]{2})\.?\s+(?P<day>\d{1,2}),?\s*(?P<year>\d{4})\s+"
    r"(?P<opponent>.*?)\s+(?P<result>[WL])\s*(?:\d*OT\s*)?"
    r"(?P<msu>\d+)-(?P<opp>\d+)(?:[ \t]+(?P<attendance>[\d,]+|[-–]))?\s*$"
)
RANK_PREFIX = re.compile(r"^(?:#|no\.)\s*(\d{1,2})(?:/\d{1,2})?\s+", re.IGNORECASE)
KICKOFF_TIME = re.compile(
    r"^(\d{1,2})(?::(\d{2}))?\s*([AP])\.?M\.?(?:\s*ET)?$", re.IGNORECASE
)


def schedule_url(season: int) -> str:
    return f"{MSU_SITE}/sports/football/schedule/{season}"


def stats_url(season: int) -> str:
    if season < 2018:
        return f"{LEGACY_STATS}/{season}-{season + 1}/teamcume.html"
    return f"{MSU_SITE}/sports/football/stats/{season}"


def _opponent_details(label: str) -> tuple[str, int | None]:
    label = label.strip()
    rank = RANK_PREFIX.match(label)
    if rank:
        label = label[rank.end() :]
    label = re.sub(r"\s+\([^)]*\)$", "", label)
    return label, int(rank[1]) if rank else None


def _opponent_key(label: str) -> str:
    """Normalize only spelling variants seen in the official source pair."""
    label = re.sub(r"\bst\.$", "state", label.casefold())
    key = re.sub(r"[^a-z]", "", label)
    aliases = {
        "centralmich": "centralmichigan",
        "flaatlantic": "floridaatlantic",
        "iowahawkeyes": "iowa",
        "pitt": "pittsburgh",
        "prairieview": "prairieviewam",
        "westernky": "westernkentucky",
        "westernmich": "westernmichigan",
    }
    return aliases.get(key, key)


def _stadium_kickoff(game_date: str, listed_time: str) -> pd.Timestamp | None:
    match = KICKOFF_TIME.fullmatch(listed_time.strip())
    if not match:
        return None
    hour = int(match[1]) % 12 + (12 if match[3].upper() == "P" else 0)
    minute = int(match[2] or 0)
    return pd.Timestamp.combine(
        pd.Timestamp(game_date), time(hour, minute)
    ).tz_localize(EASTERN)


def _read_schedule(path: Path, season: int) -> pd.DataFrame:
    page = html.fromstring(path.read_bytes())
    scripts = page.xpath('//script[@id="__NUXT_DATA__"]/text()')
    if len(scripts) != 1:
        raise ValueError(f"Expected one embedded schedule payload in {path}")
    payload = json.loads(scripts[0])

    # Nuxt serializes this payload as a list with integer references between objects.
    def value(index: int):
        return payload[index] if index >= 0 else None

    def object_values(reference: int) -> dict:
        return {key: value(index) for key, index in value(reference).items()}

    pinia = value(payload[0]["pinia"])
    state = value(pinia[1])
    schedule_state = value(state["schedule"])
    schedule_refs = value(schedule_state["schedules"])
    schedule = object_values(schedule_refs[f"schedules-football,{season}"])

    rows = []
    for reference in schedule["games"]:
        game = object_values(reference)
        opponent_label = value(game["opponent"]["title"]).strip()
        if (
            game["type"] == "S"
            or "spring" in opponent_label.casefold()
            or game["status"] == "C"
        ):
            continue
        opponent, opponent_rank = _opponent_details(opponent_label)
        game_date = datetime.fromisoformat(game["date"]).date().isoformat()
        outcome = {key: value(index) for key, index in game["result"].items()}
        facility = value(game["facility"]["title"]) if game["facility"] else None
        location = game["location"]
        is_east_lansing = location.casefold().startswith("east lansing,")
        site = game["location_indicator"]
        venue_source_url = schedule_url(season)
        is_spartan_stadium = facility == "Spartan Stadium" and is_east_lansing
        if season < 2018 and site == "H" and is_east_lansing and facility is None:
            # Legacy schedules list the home city without a venue field.
            is_spartan_stadium = True
        if game_date == "2010-09-18" and opponent == "Notre Dame":
            # The schedule marks this game neutral. The official box score names Spartan Stadium.
            is_spartan_stadium = True
            venue_source_url = NOTRE_DAME_2010_BOX_SCORE
        kickoff = (
            _stadium_kickoff(game_date, game["time"] or "")
            if is_spartan_stadium
            else None
        )
        rows.append(
            {
                "game_date": game_date,
                "opponent": opponent,
                "schedule_opponent": opponent_label,
                "schedule_rank": opponent_rank,
                "result": outcome["status"],
                "msu_score": int(outcome["team_score"]),
                "opponent_score": int(outcome["opponent_score"]),
                "is_conference": game["conference"],
                "location": location,
                "schedule_site": site,
                "venue": "Spartan Stadium" if is_spartan_stadium else facility,
                "venue_source_url": venue_source_url,
                "is_spartan_stadium": is_spartan_stadium,
                "kickoff_listed": game["time"],
                "kickoff_local": kickoff,
                "kickoff_hour": kickoff.hour + kickoff.minute / 60
                if kickoff is not None
                else None,
            }
        )
    return pd.DataFrame(rows)


def _attendance_count(value: str | None) -> int | None:
    if value in (None, "", "-", "–", "—", "N/A"):
        return None
    return int(value.replace(",", ""))


def _read_legacy_stats(page: html.HtmlElement) -> list[dict]:
    blocks = [block.text_content() for block in page.xpath("//pre")]
    schedule_blocks = [block for block in blocks if LEGACY_HEADER.search(block)]
    if len(schedule_blocks) != 1:
        raise ValueError("Expected one legacy cumulative attendance section")
    rows = []
    for line in schedule_blocks[0].splitlines():
        match = LEGACY_GAME.fullmatch(line)
        if not match:
            continue
        date_string = f"{match['month']} {match['day']} {match['year']}"
        opponent = re.sub(r"^(?:at|vs)\s+", "", match["opponent"])
        rows.append(
            {
                "game_date": datetime.strptime(date_string, "%b %d %Y")
                .date()
                .isoformat(),
                "stats_opponent": opponent,
                "result": match["result"],
                "msu_score": int(match["msu"]),
                "opponent_score": int(match["opp"]),
                "attendance": _attendance_count(match["attendance"]),
                "box_score_url": None,
            }
        )
    return rows


def _read_modern_stats(page: html.HtmlElement) -> list[dict]:
    tables = page.find_class("s-table--game-results")
    if len(tables) != 1:
        raise ValueError("Expected one cumulative game results table")
    headers = [cell.text_content().strip() for cell in tables[0].xpath(".//thead//th")]
    rows = []
    for row in tables[0].xpath(".//tbody/tr"):
        cells = row.xpath("./td")
        fields = dict(
            zip(headers, [cell.text_content().strip() for cell in cells], strict=True)
        )
        msu_score, opponent_score = map(int, fields["Score"].split("-"))
        box_scores = cells[headers.index("Opponent")].xpath(".//a/@href")
        rows.append(
            {
                "game_date": datetime.strptime(fields["Date"], "%m/%d/%Y")
                .date()
                .isoformat(),
                "stats_opponent": fields["Opponent"],
                "result": fields["Result"],
                "msu_score": msu_score,
                "opponent_score": opponent_score,
                "attendance": _attendance_count(fields["Attendance"]),
                "box_score_url": urljoin(MSU_SITE, box_scores[0])
                if box_scores
                else None,
            }
        )
    return rows


def load_games(season: int, schedule_path: Path, stats_path: Path) -> pd.DataFrame:
    """Return every completed non-exhibition game in a season with source provenance."""
    schedule = _read_schedule(schedule_path, season)
    stats_page = html.fromstring(stats_path.read_bytes())
    rows = (
        _read_legacy_stats(stats_page)
        if season < 2018
        else _read_modern_stats(stats_page)
    )
    stats = pd.DataFrame(rows)
    keys = ["game_date", "result", "msu_score", "opponent_score"]
    games = schedule.merge(
        stats, how="outer", on=keys, validate="one_to_one", indicator=True
    )
    unmatched = games.loc[games["_merge"].ne("both"), keys + ["_merge"]]
    if not unmatched.empty:
        raise ValueError(
            f"{season} schedule and stats disagree:\n{unmatched.to_string(index=False)}"
        )
    stats_opponents = games["stats_opponent"].map(_opponent_details)
    opponent_matches = stats_opponents.map(lambda item: _opponent_key(item[0])).eq(
        games["opponent"].map(_opponent_key)
    )
    if not opponent_matches.all():
        columns = ["game_date", "schedule_opponent", "stats_opponent"]
        raise ValueError(
            f"{season} opponent names disagree:\n"
            f"{games.loc[~opponent_matches, columns].to_string(index=False)}"
        )

    games = games.drop(columns="_merge")
    stats_rank = stats_opponents.map(lambda item: item[1])
    games["opponent_rank"] = (
        games["schedule_rank"].astype("Int64").fillna(stats_rank.astype("Int64"))
    )
    games["attendance"] = games["attendance"].astype("Int64")
    games["kickoff_local"] = pd.to_datetime(
        games["kickoff_local"], utc=True
    ).dt.tz_convert(EASTERN)
    games["schedule_url"] = schedule_url(season)
    games["stats_url"] = stats_url(season)
    games.insert(0, "season", season)
    return (
        games.drop(columns="schedule_rank")
        .sort_values("game_date")
        .reset_index(drop=True)
    )
