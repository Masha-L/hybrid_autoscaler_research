# src/controllers/reactive.py
from collections import deque


class Reactive:
    """Schedutil-style proportional controller with delayed reaction."""

    def __init__(self, low=0.4, high=0.8, step=0.1, delay=30):
        self.low = low
        self.high = high
        self.step = step
        self.history = deque([0.0] * delay, maxlen=delay)

    def update(self, t, cores, served, queue_len):
        util = served / cores if cores else 0
        past_util = self.history.popleft()
        self.history.append(util)
        if past_util > self.high:
            cores = int(cores * (1 + self.step))
        elif past_util < self.low:
            cores = max(1, int(cores * (1 - self.step)))
        return cores
