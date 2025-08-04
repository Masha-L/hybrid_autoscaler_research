import json, pathlib
from dataclasses import dataclass
@dataclass
class PredictiveML:
    a: float; b: float; cpu_per_vm: int = 4
    def update(self, t, cores, served, q):
        pred = max(1.0, self.a * (t + 1) + self.b)
        return max(1, int(pred // self.cpu_per_vm))
def load_from_json(path):
    coeffs = json.loads(pathlib.Path(path).read_text())
    return PredictiveML(a=coeffs["a"], b=coeffs["b"],
                        cpu_per_vm=coeffs.get("cpu_per_vm", 4))
