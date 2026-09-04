"""Pull our rated games from the public site and post-mortem the ones we did not win.

There is no API. The team page lists every rated game with a link, and each game
page embeds the PGN, clocks included, as a data: download link. This reads those
public pages -- no login, nothing written to the site -- saves each new PGN
under overnight/pgn/platform/, and runs testing.postmortem on the losses and
draws so the cross-game report stays current.

Run:  .venv\\Scripts\\python.exe -m testing.fetch_games --team "Your Team Name"
      .venv\\Scripts\\python.exe -m testing.fetch_games --team-url https://aichessathon.com/team/<id>
Add --no-analyse to only download.
"""

import argparse
import html
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SITE = "https://aichessathon.com"
OUT = Path("overnight/pgn/platform")


def get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "aichessathon-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data: bytes = response.read()
    return data.decode("utf-8", "replace")


def find_team_url(name: str) -> str:
    """The team page linked from the leaderboard under this bot name."""
    page = get(f"{SITE}/leaderboard")
    wanted = name.strip().lower()
    # Rows carry the team link as data-href and the bot name in a span; the name
    # is followed by a <small> with the display form, which is not part of it.
    for match in re.finditer(
        r'<tr data-href="(/team/[0-9a-f-]+)[^"]*"(.*?)</tr>', page, re.S
    ):
        label = re.search(r'ladder-bot-name">([^<]+)', match.group(2))
        if label and html.unescape(label.group(1)).strip().lower() == wanted:
            return SITE + match.group(1)
    raise SystemExit(f"no team named {name!r} on the leaderboard")


def list_games(team_url: str) -> list[tuple[str, str]]:
    """(game url, row text) for every game on the team page, newest first."""
    page = get(team_url)
    games: list[tuple[str, str]] = []
    for match in re.finditer(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        row = match.group(1)
        link = re.search(r'href="(/game/[0-9a-f-]+)', row)
        if link:
            text = html.unescape(re.sub(r"<[^>]+>", " ", row))
            games.append((SITE + link.group(1), " ".join(text.split())))
    return games


def fetch_pgn(game_url: str) -> str | None:
    page = get(game_url)
    match = re.search(r'href="data:application/x-chess-pgn;charset=utf-8,([^"]+)"', page)
    if not match:
        return None
    return urllib.parse.unquote(match.group(1))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch our rated games; post-mortem the losses.")
    parser.add_argument("--team", default=None, help="bot name as shown on the leaderboard")
    parser.add_argument("--team-url", default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-analyse", action="store_true")
    parser.add_argument("--all", action="store_true", help="analyse wins too")
    arguments = parser.parse_args()

    team_url: str | None = arguments.team_url
    if team_url is None and arguments.team:
        team_url = find_team_url(arguments.team)
    if not team_url:
        raise SystemExit("give --team or --team-url")
    arguments.out.mkdir(parents=True, exist_ok=True)
    games = list_games(team_url)
    print(f"{team_url}: {len(games)} games listed")

    fresh: list[Path] = []
    for index, (game_url, row) in enumerate(reversed(games), start=1):
        game_id = game_url.rsplit("/", 1)[-1][:8]
        existing = list(arguments.out.glob(f"*-{game_id}.pgn"))
        if existing:
            continue
        pgn = fetch_pgn(game_url)
        if pgn is None:
            print(f"  {game_url}: no PGN found")
            continue
        words = row.lower().split()
        colour = next((w for w in words if w in ("white", "black")), "")
        result_tag = next((w for w in words if w in ("win", "loss", "draw")), "unknown")
        opponent = ""
        if colour in words and result_tag in words:
            opponent = " ".join(words[words.index(colour) + 1 : words.index(result_tag)])
        path = arguments.out / (
            f"round-{index:02d}-{result_tag}-{colour}-vs-{slug(opponent)[:30]}-{game_id}.pgn"
        )
        path.write_text(pgn, encoding="utf-8")
        fresh.append(path)
        print(f"  saved {path.name}")
    print(f"{len(fresh)} new games")

    if arguments.no_analyse:
        return
    targets = [p for p in fresh if arguments.all or "-win-" not in p.name]
    if not targets:
        print("nothing new to analyse")
        return
    # The team page states our colour; pass it on rather than letting the
    # post-mortem guess it from move agreement, which misread round 9.
    for target in targets:
        command = [sys.executable, "-m", "testing.postmortem", str(target)]
        command += ["--out", "overnight/postmortem"]
        colour = next((c for c in ("white", "black") if f"-{c}-" in target.name), None)
        if colour:
            command += ["--colour", colour]
        print("analysing:", target.name, colour or "(side detected)")
        subprocess.run(command, check=False)


if __name__ == "__main__":
    main()
