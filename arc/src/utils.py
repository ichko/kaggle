import sys

IS_DEBUG = '--debug' in sys.argv

# torch.autograd.set_detect_anomaly(True)


def memoize(f):
    memo = {}

    def helper(*args, **kwargs):
        key = str([[str(a) for a in args], kwargs])
        if key not in memo:
            memo[key] = f(*args, **kwargs)

        return memo[key]

    return helper
