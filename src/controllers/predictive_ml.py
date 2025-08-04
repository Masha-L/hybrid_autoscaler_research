import json
import math
import pathlib
from dataclasses import dataclass


@dataclass
class PredictiveML:
    a: float
    b: float
    cpu_per_vm: int = 4  # retained for backwards compatibility

    def update(self, t, cores, served, q):
        pred = max(1.0, self.a * (t + 1) + self.b)
        # predict core demand directly and include backlog to reduce latency
        return max(1, int(math.ceil(pred + q)))


def load_from_json(path):
    coeffs = json.loads(pathlib.Path(path).read_text())
    return PredictiveML(a=coeffs["a"], b=coeffs["b"],
                        cpu_per_vm=coeffs.get("cpu_per_vm", 4))
