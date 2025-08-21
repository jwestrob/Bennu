from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple


class State(Enum):
    PLAN = auto()
    DB = auto()
    SIM = auto()
    GENOME = auto()
    ACCUM = auto()
    DECIDE = auto()
    SYN = auto()


LEGAL: Dict[State, Tuple[State, ...]] = {
    State.PLAN: (State.DB, State.SIM, State.GENOME),
    State.DB: (State.ACCUM,),
    State.SIM: (State.ACCUM,),
    State.GENOME: (State.ACCUM,),
    State.ACCUM: (State.DECIDE,),
    State.DECIDE: (State.PLAN, State.SYN),
    State.SYN: tuple(),
}


@dataclass
class FSM:
    state: State = State.PLAN

    def transition(self, to: State) -> None:
        allowed = LEGAL.get(self.state, tuple())
        if to not in allowed:
            raise ValueError(f"Illegal transition: {self.state.name} -> {to.name}")
        self.state = to

