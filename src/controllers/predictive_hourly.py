import math


class PredictiveHourly:
    def __init__(self, hourly_forecast):       # 24-value array
        self.fc = hourly_forecast

    def update(self, t, cores, served, q):
        hour = t // 3600
        # scale provision directly in cores (1 core ~= 1 req/sec) and account for
        # any queued backlog to avoid SLA violations
        return max(1, int(math.ceil(self.fc[hour] + q)))
