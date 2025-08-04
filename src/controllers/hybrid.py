import math


class Hybrid:
    def __init__(self, hourly_fc, alpha=0.7, low=0.4, high=0.8):
        self.fc, self.alpha, self.low, self.high = hourly_fc, alpha, low, high

    def update(self, t, cores, served, q):
        hour = t // 3600
        pred_core = int(math.ceil(self.fc[hour] + q))
        util = served / cores if cores else 0
        reactive = int(cores * 1.1) if util > self.high else \
                   int(cores * 0.9) if util < self.low else cores
        blended = int(self.alpha * pred_core + (1 - self.alpha) * reactive)
        return max(1, blended)
