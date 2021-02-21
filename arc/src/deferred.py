import logging
from threading import Thread
import time
from functools import wraps


# TODO: Its not cool that we spawn a thread on every call
# it will be much better with ThreadPoolExecutor
class ThreadWithReturnValue(Thread):
    @wraps(Thread)
    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def deferred(runner):
    @wraps(runner)
    def callable(*args, **kwargs):
        thread = ThreadWithReturnValue(
            target=runner,
            args=args,
            kwargs=kwargs,
        )
        thread.start()

        class Future:
            def await_result(self):
                return thread.join()

        return Future()

    return callable


@deferred
def main():
    time.sleep(1)
    print('alabala')
    return 3


if __name__ == "__main__":
    print('before')
    z = main()
    print('after')
    print(z.await_result())
