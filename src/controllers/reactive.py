# src/controllers/reactive.py
class Reactive:
    """Schedutil-style proportional controller with upper/lower thresholds."""
    def __init__(self, low=0.4, high=0.8, step=0.1):
        self.low  = low
        self.high = high
        self.step = step

    def update(self, t, cores, served, queue_len):
        util = served / cores if cores else 0
        if util > self.high:
            cores = int(cores * (1 + self.step))
        elif util < self.low:
            cores = max(1, int(cores * (1 - self.step)))
        return cores
