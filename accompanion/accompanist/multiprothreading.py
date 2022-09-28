"""
NOTE: Is this script used somewhere? I can't find it.
"""

import multiprocessing
import threading
import multiprocessing.sharedctypes as mtypes
from multiprocessing import Manager

class TestProcess(multiprocessing.Process):

    def __init__(self, value, ext_object):
        multiprocessing.Process.__init__(self)
        self.lock = multiprocessing.RLock()
        self.value = mtypes.Value(mtypes.ctypes.c_int, value)
        self.result = mtypes.Value(mtypes.ctypes.c_int, 0)
        self.ext_object = ext_object

    def run(self):
        # with self.lock:
        self.result.value = self.value.value
        self.value.value += 1
        # print(self.value.value)

        cnt = 0
        while cnt < 10:
            # with self.lock:
            self.ext_object.append(cnt)
            # print(self.ext_object.vals)
            cnt += 1

class DummyObject(object):
    def __init__(self):
        self.vals = manager.list()

    def append(self, x):
        self.vals.append(x)


class SillyProcess():
    def __init__(self, queue=multiprocessing.Queue()):
        self.queue = queue
        self.vals = []

    def append(self, val):
        self.queue.put(val)
        out = self.queue.get()
        if out is not None:
            self.vals.append(out)
    


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    manager = Manager()
    do = SillyProcess()
    tp = TestProcess(2, do)

    tp.start()
     

    print(tp.result)

    with tp.lock:
        print(tp.result)
        print(do.vals)


    print(do.vals)

         
    # tp.join()
