import math
import numpy as np
from collections import deque
from statsmodels.tsa.arima.model import ARIMA


class PredictiveARIMA:
    STEP, WIN = 60, 1800

    def __init__(self):
        self.buf = deque(maxlen=self.WIN)
        self.next_t = 0
        self.pred = 1.0
        self.prev_q = 0

    def update(self, t, cores, served, q):
        # approximate true incoming load = served + delta(queue)
        load = served + max(0, q - self.prev_q)
        self.buf.append(load)
        self.prev_q = q
        if t >= self.next_t and len(self.buf) >= 300:
            arr = np.array(self.buf, float)
            try:
                self.pred = ARIMA(arr, order=(1, 0, 1)).fit(disp=False).forecast()[0]
            except Exception:
                self.pred = arr.mean()
            self.next_t = t + self.STEP
        # provision in cores for predicted load plus any backlog
        return max(1, int(math.ceil(self.pred + q)))
