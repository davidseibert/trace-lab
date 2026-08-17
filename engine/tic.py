"""Tic-tac-toe ground truth for the engine — a faithful port of the web's
solved game (web/src/lib/tictac/game.ts) plus the pieces of the arena that the
RL loop and its evaluation need (players.ts movePrompt, metrics.ts probe
suites).

Parity is the point, not convenience: the probe/block suites are generated
through a bit-for-bit port of the web's mulberry32 PRNG and consume draws in
the exact same order, so `build_probe_suite(seed)` here names the SAME
positions as `buildProbeSuite(seed)` in the browser — a report card computed
in Python is comparable, number for number, with one computed by the Tic·arena
UI. Any deliberate divergence is called out in a comment at the spot.

Cells are indexed 0..8 row-major (0 1 2 / 3 4 5 / 6 7 8); 0 empty, 1 X, 2 O;
X moves first.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

Board = tuple[int, ...]  # length 9; immutable so it can key caches directly

# The 8 win lines: 3 rows, 3 columns, 2 diagonals.
LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

# D4 acting on cells: TRANSFORMS[g][i] = image of cell i. Same order as the
# web table (identity, three CW rotations, four mirrors).
TRANSFORMS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (2, 5, 8, 1, 4, 7, 0, 3, 6),
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
    (6, 3, 0, 7, 4, 1, 8, 5, 2),
    (2, 1, 0, 5, 4, 3, 8, 7, 6),
    (6, 7, 8, 3, 4, 5, 0, 1, 2),
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
    (8, 5, 2, 7, 4, 1, 6, 3, 0),
)

EMPTY: Board = (0,) * 9


def transform_board(g: int, b: Board) -> Board:
    t = TRANSFORMS[g]
    out = [0] * 9
    for i in range(9):
        out[t[i]] = b[i]
    return tuple(out)


def transform_moves(g: int, moves: list[int]) -> list[int]:
    t = TRANSFORMS[g]
    return [t[m] for m in moves]


def board_key(b: Board) -> int:
    """Base-3 integer encoding, 0..3^9-1 — the memo/dedupe key."""
    k = 0
    for i in range(8, -1, -1):
        k = k * 3 + b[i]
    return k


def to_move(b: Board) -> int:
    """Whose turn: X (1) when mark counts are equal."""
    x = sum(1 for c in b if c == 1)
    o = sum(1 for c in b if c == 2)
    return 1 if x == o else 2


def winner(b: Board) -> int:
    for a, m, z in LINES:
        if b[a] != 0 and b[a] == b[m] and b[a] == b[z]:
            return b[a]
    return 0


def is_terminal(b: Board) -> bool:
    return winner(b) != 0 or all(c != 0 for c in b)


def legal_moves(b: Board) -> list[int]:
    if winner(b) != 0:
        return []
    return [i for i in range(9) if b[i] == 0]


def board_from_moves(moves: list[int]) -> Board:
    """Replay a move list from the empty board; raises on any illegal move."""
    b = list(EMPTY)
    for m in moves:
        if not (0 <= m <= 8):
            raise ValueError(f"bad move {m}")
        if b[m] != 0:
            raise ValueError(f"cell {m} already taken")
        if winner(tuple(b)) != 0:
            raise ValueError(f"move {m} after game over")
        b[m] = to_move(tuple(b))
    return tuple(b)


@dataclass(frozen=True)
class Analysis:
    value: int          # minimax value from X's perspective: 1 / 0 / -1
    optimal: tuple[int, ...]  # every move achieving `value` (empty at terminal)


@lru_cache(maxsize=None)  # 5,478 legal positions — nothing
def analyze(b: Board) -> Analysis:
    w = winner(b)
    if w != 0:
        return Analysis(1 if w == 1 else -1, ())
    if all(c != 0 for c in b):
        return Analysis(0, ())
    mover = to_move(b)
    moves = legal_moves(b)
    values = []
    for m in moves:
        child = list(b)
        child[m] = mover
        values.append(analyze(tuple(child)).value)
    best = max(values) if mover == 1 else min(values)
    return Analysis(best, tuple(m for m, v in zip(moves, values) if v == best))


# ---------------------------------------------------------------------------
# mulberry32 — bit-for-bit port of web/src/lib/llm/rng.ts, so seeded suites
# match the browser's exactly. All arithmetic mod 2^32; JS's float division
# and Python's produce the same IEEE doubles.
# ---------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def mulberry32(seed: int) -> Callable[[], float]:
    a = seed & M32

    def rng() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (a | 1)) & M32
        t = (((t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32) ^ t) & M32
        return (t ^ (t >> 14)) / 4294967296

    return rng


def rand_int(rng: Callable[[], float], n: int) -> int:
    return int(rng() * n)


def generate_game(rng: Callable[[], float], policy: Literal["optimal", "random"]) -> list[int]:
    """One full game to a terminal position; draw order matches game.ts."""
    b = list(EMPTY)
    moves: list[int] = []
    while not is_terminal(tuple(b)):
        pool = list(analyze(tuple(b)).optimal) if policy == "optimal" else legal_moves(tuple(b))
        m = pool[rand_int(rng, len(pool))]
        b[m] = to_move(tuple(b))
        moves.append(m)
    return moves


# ---------------------------------------------------------------------------
# Probe suites — port of metrics.ts buildProbeSuite / buildBlockSuite
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbePosition:
    moves: tuple[int, ...]
    board: Board
    legal: tuple[int, ...]
    optimal: tuple[int, ...]


SUITE_SIZE = 40
EQUIV_SUB = 12


def build_probe_suite(seed: int, n: int = SUITE_SIZE) -> list[ProbePosition]:
    """Seeded, deduped non-terminal positions over plies 0..7 from mixed-policy
    games. Consumes rng draws in the same order as the TS version (policy coin,
    game moves, ply cut), so a given seed names identical positions."""
    rng = mulberry32(seed + 1000)
    seen: set[int] = set()
    out: list[ProbePosition] = []
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        policy: Literal["optimal", "random"] = "optimal" if rng() < 0.5 else "random"
        game = generate_game(rng, policy)
        cut = rand_int(rng, min(len(game), 8))
        moves = game[:cut]
        board = board_from_moves(moves)
        if is_terminal(board):
            continue
        key = board_key(board)
        if key in seen:
            continue
        seen.add(key)
        a = analyze(board)
        out.append(ProbePosition(tuple(moves), board, tuple(legal_moves(board)), a.optimal))
    return out


def is_forced_block(p: ProbePosition) -> bool:
    """The optimal move is UNIQUE and completes a block of an opponent
    two-in-a-row — the report card's tactical predicate."""
    if len(p.optimal) != 1:
        return False
    opp = 2 if to_move(p.board) == 1 else 1
    return any(
        p.optimal[0] in line and sum(1 for c in line if p.board[c] == opp) == 2
        for line in LINES
    )


def build_block_suite(seed: int, n: int = 24) -> list[ProbePosition]:
    return [p for p in build_probe_suite(seed, 160) if is_forced_block(p)][:n]


# ---------------------------------------------------------------------------
# The arena prompt — port of players.ts movePrompt (raw mode), verbatim, so a
# checkpoint trained here reads the exact text the Tic·arena report card uses.
# ---------------------------------------------------------------------------

CELL_LEGEND = "0 1 2\n3 4 5\n6 7 8"


def render_board_text(b: Board) -> str:
    mark = {0: ".", 1: "X", 2: "O"}
    return "\n".join(
        f"{mark[b[r]]} {mark[b[r + 1]]} {mark[b[r + 2]]}" for r in (0, 3, 6)
    )


def move_prompt(moves: list[int], chat: bool = False) -> str:
    b = board_from_moves(moves)
    mover = "X" if to_move(b) == 1 else "O"
    base = (
        f"Tic-tac-toe. Cells are numbered 0-8, left to right, top to bottom:\n{CELL_LEGEND}\n"
        f"Current board (X, O, . = empty):\n{render_board_text(b)}\n"
        f"It is {mover}'s turn."
    )
    if chat:
        return f"{base}\nReply with only the digit of the best cell for {mover} to play."
    return f"{base} The single best cell number for {mover} to play is"
