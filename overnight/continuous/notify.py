"""Email Toby when a version is ready to review (or on a pipeline event).

Reuses the Quant Role Scout SMTP setup (C:/dev/quant-role-scout/config.json ->
emailer.send). The message carries: the bot's platform record so far, the version's
change list and measured improvement (from overnight/continuous/CANDIDATE.md), and
where the zip is.

  .venv/Scripts/python.exe -m overnight.continuous.notify --candidate   # CANDIDATE.md
  .venv/Scripts/python.exe -m overnight.continuous.notify --text "message"    # a plain note
  .venv/Scripts/python.exe -m overnight.continuous.notify --test              # "pipeline online"
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOUT = Path(r"C:/dev/quant-role-scout")
CANDIDATE = ROOT / "overnight/continuous/CANDIDATE.md"
PGN_DIR = ROOT / "overnight/pgn/platform"
TEAM = "make_no_mistakes"


def platform_record() -> tuple[str, list[str]]:
    """W-D-L over the rated platform games we have downloaded, plus the last five."""
    games: list[tuple[int, str, str, str]] = []
    for pgn in sorted(PGN_DIR.glob("*.pgn")):
        text = pgn.read_text(encoding="utf-8", errors="replace")
        headers = dict(re.findall(r'\[(\w+) "([^"]*)"\]', text))
        white, black, result = (
            headers.get("White", ""),
            headers.get("Black", ""),
            headers.get("Result", "*"),
        )
        if result == "*":
            continue
        ours_white = TEAM in white.lower().replace(" ", "_")
        opponent = black if ours_white else white
        score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)
        if score is None:
            continue
        ours = score if ours_white else 1.0 - score
        rnd = re.search(r"round-(\d+)", pgn.name)
        games.append(
            (
                int(rnd.group(1)) if rnd else 0,
                "win" if ours == 1 else "draw" if ours == 0.5 else "loss",
                opponent,
                "white" if ours_white else "black",
            )
        )
    games.sort()
    tally = Counter(g[1] for g in games)
    record = f"{tally['win']}-{tally['draw']}-{tally['loss']} (W-D-L) over {len(games)} rated games"
    if games:
        pts = tally["win"] + 0.5 * tally["draw"]
        record += f", {100 * pts / len(games):.0f}%"
    last = [f"round {r}: {res} as {col} vs {opp}" for r, res, opp, col in games[-5:]]
    return record, last


def send_email(subject: str, body_html: str) -> bool:
    sys.path.insert(0, str(SCOUT))
    import emailer  # type: ignore[import-not-found]

    cfg = json.loads((SCOUT / "config.json").read_text(encoding="utf-8"))["email"]
    if not emailer.email_configured(cfg):
        print("email not configured in the scout config", file=sys.stderr)
        return False
    return bool(emailer.send(cfg, subject, body_html, cfg.get("to") or [cfg["smtp_user"]]))


def render(title: str, sections: list[tuple[str, str]]) -> str:
    parts = [f"<h2 style='font-family:sans-serif'>{html.escape(title)}</h2>"]
    for heading, text in sections:
        body = html.escape(text).replace("\n", "<br>")
        parts.append(
            f"<h3 style='font-family:sans-serif'>{html.escape(heading)}</h3>"
            f"<p style='font-family:monospace;white-space:pre-wrap'>{body}</p>"
        )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", action="store_true", help="send CANDIDATE.md as a ready-to-ship notice"
    )
    parser.add_argument("--text", default="", help="send a plain note")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--subject", default="")
    arguments = parser.parse_args()
    record, last = platform_record()
    platform = record + ("\n" + "\n".join(last) if last else "")
    if arguments.candidate:
        if not CANDIDATE.exists():
            raise SystemExit("no CANDIDATE.md")
        text = CANDIDATE.read_text(encoding="utf-8")
        version = re.search(r"^#\s*(.+)$", text, re.M)
        title = version.group(1).strip() if version else "candidate ready"
        subject = arguments.subject or f"[Chessathon] {title} ready to review"
        sections = [("Platform record (make_no_mistakes)", platform), ("This version", text)]
    elif arguments.test:
        subject = arguments.subject or "[Chessathon] improvement pipeline online"
        sections = [
            ("Platform record (make_no_mistakes)", platform),
            (
                "Note",
                "The recursive research/build/test loop is running. You will get one "
                "email per version that passes its gate, with the change list, the "
                "measured gain and the zip location.",
            ),
        ]
    else:
        subject = arguments.subject or "[Chessathon] note"
        sections = [("Platform record (make_no_mistakes)", platform), ("Note", arguments.text)]
    ok = send_email(subject, render(subject, sections))
    print("sent" if ok else "not sent", subject)


if __name__ == "__main__":
    main()
