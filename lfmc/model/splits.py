import hashlib

from frozendict import frozendict

from lfmc.model.mode import Mode

SPLIT_TO_HEX = frozendict(
    {
        0: frozenset(["0", "1", "2", "3"]),
        1: frozenset(["4", "5", "6", "7"]),
        2: frozenset(["8", "9", "A", "B"]),
        3: frozenset(["C", "D", "E", "F"]),
    }
)


def num_splits() -> int:
    return len(SPLIT_TO_HEX)


def get_mode_from_hex(sorting_id: int, split_id: int) -> Mode:
    if split_id < 0 or split_id >= num_splits():
        raise ValueError(f"Split ID must be in range [0, {num_splits() - 1}], got {split_id}")
    hex_value = hashlib.sha256(str(sorting_id).encode("utf-8")).hexdigest()[0].upper()
    return Mode.VALIDATION if hex_value in SPLIT_TO_HEX[split_id] else Mode.TRAIN
