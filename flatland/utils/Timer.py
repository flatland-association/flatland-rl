from timeit import default_timer


class Timer(object):
    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0

    def start(self):
        self.start_time = default_timer()

    def end(self):
        self.total_time += self.get_current()
        return self.get_total_time()

    def get_total_time(self):
        return self.total_time

    def get_current(self):
        return default_timer() - self.start_time

    def reset(self):
        self.__init__()
