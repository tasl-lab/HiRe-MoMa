import time
from collections import defaultdict


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        self.start_times[key] = time.time()

    def tock(self, key):
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times.pop(key)

    def context(self, key):
        timer = self

        class _Ctx:
            def __enter__(self):
                timer.tick(key)

            def __exit__(self, *args):
                timer.tock(key)

        return _Ctx()

    def get_average_times(self, reset=True):
        ret = {k: self.times[k] / self.counts[k] for k in self.counts}
        if reset:
            self.reset()
        return ret
