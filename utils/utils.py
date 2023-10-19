import sys
from time import perf_counter

class timer(object):
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, *args):
        self.time = perf_counter() - self.time


def export_best(res, obj_sample, t):
    out = sys.stdout
    with open('times2.out', 'a') as f:
        sys.stdout = f
        print(f"{t}")

    with open('best_accs2.out', 'a') as f:
        sys.stdout = f
        print(f'{-res.max().log()}')

    with open('best_configs2.out', 'a') as f:
        sys.stdout = f
        print(f'{obj_sample[res.argmax()].tolist()}')
    
    sys.stdout = out
    sys.exit()