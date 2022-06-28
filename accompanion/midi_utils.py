import threading

class VirtualMidiThroughPort(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def send(self, msg):
        self.outport.send(msg)
