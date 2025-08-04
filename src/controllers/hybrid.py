import math


class Hybrid:
    def __init__(self, hourly_fc, buffer=0.1, low=0.4, high=0.8):
        self.fc = hourly_fc
        self.buffer = buffer
        self.low = low
        self.high = high

    def update(self, t, cores, served, q):
        hour = t // 3600
        pred_core = self.fc[hour] + q
        util = served / cores if cores else 0
        reactive = cores
        if util > self.high:
            reactive = cores * 1.1
        elif util < self.low:
            reactive = cores * 0.9
        blended = max(pred_core, reactive)
        return max(1, int(math.ceil(blended * (1 + self.buffer))))
