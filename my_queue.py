from multiprocessing import Pipe, Lock



class MyQueue(object):

    def __init__(self):
        """ Initialize MyQueue and it's members.
        """
        self.reader, self.writer = Pipe()
        self.lock_w = Lock()
        self.lock_r = Lock()

    def put(self, msg):
        """Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        """
        self.lock_w.acquire()
        self.writer.send(msg)
        self.lock_w.release()

    def get(self):
        """Get the next message from queue (FIFO)

        Return
        ------
        An object
        """
        self.lock_r.acquire()
        res = self.reader.recv()
        self.lock_r.release()
        return res







