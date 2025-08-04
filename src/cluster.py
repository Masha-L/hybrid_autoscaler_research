# src/cluster.py
import pandas as pd
import numpy as np
from collections import deque

class Cluster:
    """Minimal 1-sec-tick simulator for CPU-scaling evaluation."""
    def __init__(self, trace: pd.Series, controller, vms=100, cpu_per_vm=4):
        self.trace        = trace.values          # requests / sec
        self.times        = trace.index           # pandas DatetimeIndex
        self.ctrl         = controller
        self.vms          = vms
        self.cpu_per_vm   = cpu_per_vm
        self.active_cores = vms * cpu_per_vm
        # metrics
        self.latencies    = []
        self.wasted_cpu_s = 0
        self.sla_viol     = 0
        self.queue        = deque()               # (arrival_time) per request

    def step(self, idx):
        load = int(self.trace[idx])               # incoming requests
        now  = idx                                # integer second
        # enqueue new arrivals
        for _ in range(load):
            self.queue.append(now)
        # how many can we serve this tick?
        served = min(len(self.queue), self.active_cores)
        for _ in range(served):
            arrive = self.queue.popleft()
            latency = (now - arrive) * 1000 + 8   # 8 ms service time
            self.latencies.append(latency)
            if latency > 200:
                self.sla_viol += 1
        # wasted cores
        self.wasted_cpu_s += (self.active_cores - served)
        # controller decides scaling action
        self.active_cores = self.ctrl.update(now, self.active_cores,
                                             served, len(self.queue))

    def run(self):
        for t in range(len(self.trace)):
            self.step(t)
        return self._summary()

    def _summary(self):
        lat = np.array(self.latencies)
        return {
            "latency_ms":   lat.mean() if lat.size else 0,
            "throughput":   len(lat) / len(self.trace),
            "sla_violation": self.sla_viol / len(lat) * 100 if lat.size else 0,
            "wasted_cpu_h": self.wasted_cpu_s / 3600
        }
