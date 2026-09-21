from typing import Optional

from sumo_rl.environment.traffic_signal import TrafficSignal


class AllRedTrafficSignal(TrafficSignal):
    """TrafficSignal that inserts an all-red clearance phase between yellow and the next green.

    State machine for agent-controlled mode becomes:
        green(p1) -> yellow -> all-red -> green(p2)
    instead of the original:
        green(p1) -> yellow -> green(p2)

    fixed_ts mode is untouched functionally (SUMO's native program already
    plays back whatever phases -- including all-red -- are defined in the
    .net.xml), we only fix the green-phase count here.
    """

    def __init__(
        self,
        env,
        ts_id,
        delta_time,
        yellow_time,
        red_time,
        min_green,
        max_green,
        enforce_max_green,
        begin_time,
        reward_fn,
        reward_weights,
        sumo,
        program_id: Optional[str] = None,
    ):
        self.red_time = red_time
        self.is_red = False
        self._pending_transition = None
        self.program_id = program_id
        super().__init__(
            env,
            ts_id,
            delta_time,
            yellow_time,
            min_green,
            max_green,
            enforce_max_green,
            begin_time,
            reward_fn,
            reward_weights,
            sumo,
        )

    def _get_program_logic(self):
        """Return the Logic object to read phases from, selecting by programID if one was given.

        Falls back to the first program SUMO loaded (index 0) when
        self.program_id is None, matching upstream sumo-rl's behavior.
        """
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        if self.program_id is None:
            return programs[0]
        for p in programs:
            if p.programID == self.program_id:
                return p
        available = [p.programID for p in programs]
        raise ValueError(
            f"programID {self.program_id!r} not found for traffic light {self.id!r}. "
            f"Available programIDs: {available}"
        )

    def _build_phases(self):
        logic_src = self._get_program_logic()
        phases = logic_src.phases

        def _is_green(state):
            return "y" not in state and (state.count("r") + state.count("s") != len(state))

        if self.env.fixed_ts:
            # Explicitly select this program as the one SUMO drives natively.
            # Needed because the net file may define more than one <tlLogic>
            # for this id, and we never call setProgramLogic in this branch
            # for SUMO to infer our intent from -- fixed_ts just lets SUMO's
            # own scheduler play back whatever program is "active".
            self.sumo.trafficlight.setProgram(self.id, logic_src.programID)
            self.num_green_phases = sum(1 for p in phases if _is_green(p.state))
            return

        self.green_phases = [self.sumo.trafficlight.Phase(60, p.state) for p in phases if _is_green(p.state)]
        if not self.green_phases:
            raise ValueError(
                f"No green phases found for traffic light {self.id!r} in program "
                f"{logic_src.programID!r}. Every phase state contained 'y' or was fully red/stop."
            )
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()
        self.yellow_dict = {}
        self.red_dict = {}

        all_red_state = "r" * len(self.green_phases[0].state)

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue

                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]

                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

                if self.red_time > 0:
                    self.red_dict[(i, j)] = len(self.all_phases)
                    self.all_phases.append(self.sumo.trafficlight.Phase(self.red_time, all_red_state))

        logic_src.type = 0
        logic_src.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic_src)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    def update(self):
        """Advance the yellow -> [all-red] -> green state machine."""
        self.time_since_last_phase_change += 1

        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            self.is_yellow = False
            if self.red_time > 0:
                self.is_red = True
                self.sumo.trafficlight.setRedYellowGreenState(
                    self.id, self.all_phases[self.red_dict[self._pending_transition]].state
                )
            else:
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)

        elif self.is_red and self.time_since_last_phase_change == self.yellow_time + self.red_time:
            self.is_red = False
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)

    def set_next_phase(self, new_phase: int):
        """Sets what will be the next green phase, inserting yellow + all-red if changing.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        clearance_time = self.yellow_time + self.red_time

        if self.enforce_max_green and new_phase == self.green_phase and self.time_since_last_phase_change >= self.max_green:
            new_phase = (self.green_phase + 1) % self.num_green_phases

        if self.green_phase == new_phase or self.time_since_last_phase_change < clearance_time + self.min_green:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # Compute the transition key BEFORE overwriting green_phase, same
            # as upstream does for yellow_dict -- update() needs it later to
            # look up the matching red state.
            self._pending_transition = (self.green_phase, new_phase)
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[self._pending_transition]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
