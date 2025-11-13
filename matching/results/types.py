from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

CallKey = Tuple[str, int]                     # (block_label, call_index)
BindingKey = Tuple[str, Tuple[Tuple[str,str], ...]]

@dataclass(frozen=True)
class Interval:
    t0: int
    t1: int  # inclusive

@dataclass
class PerCallSignal:
    segment_id: str
    roles: Dict[str, str]            # partial assignment (e.g., {'ego_vehicle':'veh_12'} or both roles)
    T: int
    intervals: List[Interval]
    mask: Optional[np.ndarray] = None

    # NEW: needed by the block combiners
    call_index: int = -1             # which call (index in calls list) produced this signal
    roles_used: Tuple[str, ...] = () # roles referenced by this call (ego + any 'same_as'/'reference' etc.)

@dataclass
class BlockSignal:
    segment_id: str
    roles: Dict[str, str]
    T: int
    intervals: List[Interval]
    mask: Optional[np.ndarray] = None

@dataclass
class ResultsStore:
    by_call: Dict[CallKey, List[PerCallSignal]] = field(default_factory=dict)

    def add(self, call_key: CallKey, signal: PerCallSignal) -> None:
        self.by_call.setdefault(call_key, []).append(signal)

    def signals(self, call_key: CallKey) -> List[PerCallSignal]:
        return self.by_call.get(call_key, [])