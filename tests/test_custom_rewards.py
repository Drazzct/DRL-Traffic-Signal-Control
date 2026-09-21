import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_drl.environment.custom_rewards import CombinedReward


class FakeLane:
    def __init__(self, vehicle_ids, halting_number, mean_speed):
        self._vehicle_ids = vehicle_ids
        self._halting_number = halting_number
        self._mean_speed = mean_speed

    def getLastStepVehicleIDs(self, _lane):
        return self._vehicle_ids

    def getLastStepHaltingNumber(self, _lane):
        return self._halting_number

    def getLastStepMeanSpeed(self, _lane):
        return self._mean_speed


class FakeVehicle:
    def __init__(self):
        self.waiting_times = {}

    def getAccumulatedWaitingTime(self, veh):
        return self.waiting_times[veh]

    def getVehicleClass(self, veh):
        return "passenger"


class FakeSUMO:
    def __init__(self, lanes, waiting_times):
        self.lane = FakeLaneCollection(lanes)
        self.vehicle = FakeVehicle()
        self.vehicle.waiting_times = waiting_times


class FakeLaneCollection:
    def __init__(self, lanes):
        self._lanes = lanes

    def getLastStepVehicleIDs(self, lane):
        return self._lanes[lane]["vehicles"]

    def getLastStepHaltingNumber(self, lane):
        return self._lanes[lane]["queue"]

    def getLastStepMeanSpeed(self, lane):
        return self._lanes[lane]["speed"]

    def getLength(self, lane):
        return self._lanes[lane].get("length", 100.0)


class FakeTrafficSignal:
    def __init__(self, lanes, waiting_times):
        self.lanes = list(lanes.keys())
        self.sumo = FakeSUMO(lanes, waiting_times)
        self.green_phase = 0
        self.time_on_phase = 0.0


def test_combined_reward_normalizes_wait_and_queue_components():
    lanes = {
        "lane1": {"vehicles": ["veh1"], "queue": 4, "speed": 0.0},
        "lane2": {"vehicles": ["veh2"], "queue": 2, "speed": 0.0},
    }
    waiting_times = {"veh1": 1.0, "veh2": 2.0}
    reward_fn = CombinedReward(wait_weight=1.0, queue_weight=1.0)

    first_reward = reward_fn(FakeTrafficSignal(lanes, waiting_times))

    lanes = {
        "lane1": {"vehicles": ["veh1"], "queue": 1, "speed": 0.0},
        "lane2": {"vehicles": ["veh2"], "queue": 1, "speed": 0.0},
    }
    waiting_times = {"veh1": 1.0, "veh2": 0.5}
    second_reward = reward_fn(FakeTrafficSignal(lanes, waiting_times))

    # max_wait = 2400.0, max_queue = 194.0
    expected_first = -(3.0 / 2400.0) - (6.0 / 194.0)
    expected_second = -(1.5 / 2400.0) - (2.0 / 194.0)
    
    assert abs(first_reward - expected_first) < 1e-6
    assert abs(second_reward - expected_second) < 1e-6


def test_combined_reward_flicker_penalty():
    lanes = {
        "lane1": {"vehicles": ["veh1"], "queue": 1, "speed": 10.0},
    }
    waiting_times = {"veh1": 1.0}
    reward_fn = CombinedReward(wait_weight=0.0, queue_weight=0.0, flicker_weight=1.0, deadlock_penalty=0.0, lambda_f=0.9)
    
    signal = FakeTrafficSignal(lanes, waiting_times)
    signal.green_phase = 0
    r1 = reward_fn(signal)
    
    # First step, phase hasn't changed from last_phase (since last_phase becomes 0)
    assert r1 == 0.0
    
    signal.green_phase = 1
    r2 = reward_fn(signal)
    
    # Phase changed, phase_changed = 1.0. flicker_state = 0.9 * 0.0 + 1.0 = 1.0
    # f_norm = 1.0 / (1.0 - 0.9) = 10.0
    # p_flicker = 1.0 * (1.0 / 10.0) = 0.1, r_t = -0.1
    assert abs(r2 - (-0.1)) < 1e-6


def test_combined_reward_deadlock_penalty():
    lanes = {
        "lane1": {"vehicles": ["veh1"], "queue": 1, "speed": 0.05}, # speed < v_stall (0.1)
    }
    waiting_times = {"veh1": 1.0}
    reward_fn = CombinedReward(wait_weight=0.0, queue_weight=0.0, flicker_weight=0.0, deadlock_penalty=5.0, v_stall=0.1, t_stall_limit=10.0, delta_time=5.0)
    
    signal = FakeTrafficSignal(lanes, waiting_times)
    r1 = reward_fn(signal) # stall_time = 5.0 <= 10.0 -> p_deadlock = 0
    assert r1 == 0.0
    
    r2 = reward_fn(signal) # stall_time = 10.0 <= 10.0 -> p_deadlock = 0
    assert r2 == 0.0
    
    r3 = reward_fn(signal) # stall_time = 15.0 > 10.0 -> p_deadlock = 5.0 -> r_t = -5.0
    assert r3 == -5.0
    
    # Now speed goes up
    lanes["lane1"]["speed"] = 2.0
    r4 = reward_fn(signal) # stall_time = 0.0 -> p_deadlock = 0
    assert r4 == 0.0
