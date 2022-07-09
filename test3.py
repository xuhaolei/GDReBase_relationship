import random

class A:
    def __init__(self):
        self.g = iter(range(10001))
        self.batch = 100
        self.last_set = None

    def __iter__(self):
        return self

    def __next__(self):
        # make pep8 happy
        res = []
        try:
            res = []
            for _ in range(self.batch):
                res.append(next(self.g))
            # res = [next(self.g) for _ in range(self.batch)]
            random.shuffle(res)
            return tuple(res)
        except StopIteration:
            random.shuffle(res)
            self.last_set = tuple(res)
            raise StopIteration

a = A()
for _r in a:
    print(_r)
print(a.last_set)
