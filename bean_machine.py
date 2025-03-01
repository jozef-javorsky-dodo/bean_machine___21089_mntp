from __future__ import annotations
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from random import gauss, random
from typing import Final, List, Tuple, TypeAlias, Iterator
from PIL import Image, ImageDraw

Position: TypeAlias = float
Frequency: TypeAlias = int
Color: TypeAlias = Tuple[int, int, int]

@dataclass(frozen=True)
class BoardConfig:
    NUM_ROWS: Final[int] = 12
    NUM_BALLS: Final[int] = 100_000
    BOARD_WIDTH: Final[int] = 700
    BOARD_HEIGHT: Final[int] = 500
    PEG_RADIUS: Final[int] = 4
    BACKGROUND_COLOR: Final[Color] = (102, 51, 153)
    LEFT_COLOR: Final[Color] = (122, 122, 244)
    RIGHT_COLOR: Final[Color] = (122, 244, 122)
    SMOOTHING_WINDOW: Final[int] = 3
    DAMPING_FACTOR: Final[float] = 0.8
    ELASTICITY: Final[float] = 0.7
    INITIAL_VARIANCE: Final[float] = 2.0
    MIN_BOUNCE_PROBABILITY: Final[float] = 0.2
    MAX_BOUNCE_PROBABILITY: Final[float] = 0.8
    BOUNCE_DISTANCE_FACTOR: Final[float] = 0.1
    PROGRESS_DIVISIONS: Final[int] = 20
    BOUNCE_PROB_CACHE_SIZE: Final[int] = 128
    DEFAULT_IMAGE_FILENAME: Final[str] = "galton_board.png"
    LOG_FORMAT: Final[str] = "%(levelname)s: %(message)s"
    HALF_PEG_RADIUS_FACTOR: Final[float] = 0.5
    BOUNCE_PROB_CENTER: Final[float] = 0.5
    HISTOGRAM_BAR_MIN_WIDTH: Final[int] = 1

@dataclass
class GaltonBoard:
    num_rows: int = field(default=BoardConfig.NUM_ROWS)
    num_balls: int = field(default=BoardConfig.NUM_BALLS)
    board_width: int = field(default=BoardConfig.BOARD_WIDTH)
    board_height: int = field(default=BoardConfig.BOARD_HEIGHT)
    slot_counts: List[Frequency] = field(default_factory=list)
    image: Image.Image = field(init=False)
    draw: ImageDraw.Draw = field(init=False)

    def __post_init__(self) -> None:
        self.slot_counts = [0] * self.board_width
        self.image = Image.new("RGB", (self.board_width, self.board_height),
                                BoardConfig.BACKGROUND_COLOR)
        self.draw = ImageDraw.Draw(self.image)
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        dims = (self.num_rows, self.num_balls, self.board_width, self.board_height)
        if not all(d > 0 for d in dims):
            raise ValueError("Dimensions must be positive.")
        if self.board_width <= 2 * BoardConfig.PEG_RADIUS:
            raise ValueError("Board width must be greater than twice peg radius.")

    def simulate(self) -> None:
        slots = [0] * self.board_width
        progress_step = max(1, self.num_balls // BoardConfig.PROGRESS_DIVISIONS)
        for i, idx in enumerate(self._generate_ball_paths(), start=1):
            slots[idx] += 1
            if i % progress_step == 0:
                logging.info(f"Simulated {i}/{self.num_balls} balls.")
        self._apply_smoothing(slots)

    def _generate_ball_paths(self) -> Iterator[int]:
        for _ in range(self.num_balls):
            yield self._simulate_ball_path()

    def _simulate_ball_path(self) -> int:
        center = self.board_width / 2
        pos = center + gauss(0, BoardConfig.INITIAL_VARIANCE)
        momentum = 0.0
        mult = BoardConfig.PEG_RADIUS * 2
        for row in range(self.num_rows):
            pos, momentum = self._calculate_ball_step(pos, momentum, row, mult)
        return int(self._constrain_position(pos))

    def _calculate_ball_step(self, pos: Position, momentum: float, row: int,
                             mult: float) -> tuple[Position, float]:
        offset = (row % 2) * (BoardConfig.PEG_RADIUS * BoardConfig.HALF_PEG_RADIUS_FACTOR)
        peg = pos + offset
        delta = (pos - peg) / BoardConfig.PEG_RADIUS
        bp = self._calculate_bounce_probability(delta)
        direction = 1 if random() < bp else -1
        force = (1.0 - abs(delta)) * BoardConfig.ELASTICITY
        new_mom = momentum * BoardConfig.DAMPING_FACTOR + direction * force * mult
        new_pos = self._constrain_position(pos + new_mom)
        return new_pos, new_mom

    @staticmethod
    @lru_cache(maxsize=BoardConfig.BOUNCE_PROB_CACHE_SIZE)
    def _calculate_bounce_probability(delta: float) -> float:
        p = BoardConfig.BOUNCE_PROB_CENTER + BoardConfig.BOUNCE_DISTANCE_FACTOR * delta
        return max(BoardConfig.MIN_BOUNCE_PROBABILITY,
                   min(BoardConfig.MAX_BOUNCE_PROBABILITY, p))

    def _constrain_position(self, pos: Position) -> Position:
        lo = BoardConfig.PEG_RADIUS
        hi = self.board_width - BoardConfig.PEG_RADIUS
        return max(lo, min(hi, pos))

    def _apply_smoothing(self, slots: List[Frequency]) -> None:
        w = BoardConfig.SMOOTHING_WINDOW
        for i in range(len(self.slot_counts)):
            start_idx = max(0, i - w)
            end_idx = min(len(slots), i + w + 1)
            segment = slots[start_idx:end_idx]
            if (n := end_idx - start_idx) > 0:
                self.slot_counts[i] = sum(segment) // n

    def generate_image(self) -> Image.Image:
        m = max(self.slot_counts, default=0)
        if m:
            bw = max(BoardConfig.HISTOGRAM_BAR_MIN_WIDTH,
                     self.board_width // len(self.slot_counts))
            self._draw_all_bars(m, bw)
        return self.image

    def _draw_all_bars(self, max_freq: int, bw: int) -> None:
        for i, f in enumerate(self.slot_counts):
            self._draw_histogram_bar(i, f, max_freq, bw)

    def _draw_histogram_bar(self, i: int, f: int, mx: int, bw: int) -> None:
        if not mx:
            return
        bh = int(f / mx * self.board_height)
        xs = i * bw
        xe = xs + bw
        ys = self.board_height - bh
        col = self._get_bar_color(xs)
        self.draw.rectangle([xs, ys, xe, self.board_height], fill=col)

    def _get_bar_color(self, x: int) -> Color:
        return BoardConfig.LEFT_COLOR if x < self.board_width // 2 else BoardConfig.RIGHT_COLOR

    def save_image(self, filename: str = BoardConfig.DEFAULT_IMAGE_FILENAME) -> None:
        try:
            out = Path(filename).resolve()
            self.generate_image().save(out)
        except (IOError, OSError) as e:
            logging.error(f"Failed to save image to {filename}: {e}.")
            raise

def generate_galton_board() -> None:
    gb = GaltonBoard()
    gb.simulate()
    gb.save_image()

def main() -> None:
    logging.basicConfig(level=logging.INFO, format=BoardConfig.LOG_FORMAT)
    try:
        generate_galton_board()
    except Exception as e:
        logging.exception(f"Fatal error: {e}.")
        raise

if __name__ == "__main__":
    main()