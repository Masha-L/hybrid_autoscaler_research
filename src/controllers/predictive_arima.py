import numpy as np
from collections import deque
from statsmodels.tsa.arima.model import ARIMA
class PredictiveARIMA:
    CPU_PER_VM, STEP, WIN = 4, 60, 1800
    def __init__(self):
        self.buf, self.next_t, self.pred = deque(maxlen=self.WIN), 0, 1
    def update(self, t, cores, served, q):
        self.buf.append(served)
        if t >= self.next_t and len(self.buf) >= 300:
            arr = np.array(self.buf, float)
            try:  self.pred = ARIMA(arr, order=(1,0,1)).fit(disp=False).forecast()[0]
            except: self.pred = arr.mean()
            self.next_t = t + self.STEP
        return max(1, int(max(1.0, self.pred) // self.CPU_PER_VM))
