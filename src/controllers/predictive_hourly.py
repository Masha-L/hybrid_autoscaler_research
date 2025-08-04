class PredictiveHourly:
    CPU_PER_VM = 4
    def __init__(self, hourly_forecast):       # 24-value array
        self.fc = hourly_forecast
    def update(self, t, cores, served, q):
        hour = t // 3600
        return max(1, int(self.fc[hour] // self.CPU_PER_VM))
