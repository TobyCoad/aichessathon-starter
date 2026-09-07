"""Event constants, as shipped with the starter kit. Canonical source:
https://aichessathon.com/docs/rules.md

DO NOT "correct" these. AGENTS.md says not to edit harness/ -- it mirrors the platform's
protocol and clock -- and an earlier attempt to fix the two numbers below made things
worse rather than better:

  * PLY_CAP. `testing/referee.py` argues the real cap is 600 and that a game reaching it
    is a draw. But `harness/referee.py:57` awards the game on raw material at whatever
    this constant says, so raising it moves the material award rather than removing it;
    `agent.py` ships ADJ_V2 = False, so its own adjudication planning is tuned to
    ADJUDICATION_PLY = 300, and raising it here alone puts engine and referee 300 plies
    apart. There is also direct counter-evidence: our own platform game
    round-18-loss-black-vs-pheanup terminated "adjudication" at EXACTLY ply 300.
  * INIT_BUDGET_S. Platform imports of 74.1 s and 88.1 s both played their games and
    only >90 s forfeited, so 60 is probably conservative -- and INIT_ASYNC deliberately
    returns at INIT_READY_S = 72.0. A local `make arena` can therefore score a healthy
    engine as an init failure. That is a known local artefact, not a bug to fix here.

`testing/referee.py` is the corrected referee and is what every measurement that matters
should use. Change these numbers only if the rules page itself changes.
"""

INIT_BUDGET_S = 60.0
BASE_MS = 120_000
INCREMENT_MS = 500
PLY_CAP = 300
STDOUT_CAP = 4096
WATCHDOG_GRACE_MS = 500
