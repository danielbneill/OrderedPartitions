import multiprocessing
import proto

class OptimizerSWIG(object):
    ''' Task-based C++ optimizer
    '''
    def __init__(self, num_partitions, g, h, sweep_mode=False):
        self.N = len(g)
        self.num_partitions = num_partitions
        self.g_c = proto.GMPArray()
        self.h_c = proto.GMPArray()
        self.g_c = g
        self.h_c = h

        self.sweep_mode = sweep_mode

    def __call__(self):
        if self.sweep_mode:
            return proto.sweep_parallel__DP(self.N, self.num_partitions, self.g_c, self.h_c)
        else:
            return proto.optimize_one__DP(self.N, self.num_partitions, self.g_c, self.h_c)

class EndTask(object):
    pass

class OptimizerTask(object):
    def __init__(self, N, num_partitions, g, h):
        g_c = proto.GMPArray()
        h_c = proto.GMPArray()        
        g_c = g
        h_c = h
        self.task = partial(self._task, N, num_partitions, g_c, h_c)

    def __call__(self):
        return self.task()
        
    @staticmethod
    def _task(N, num_partitions, g, h):
        s, w = proto.optimize_one__DP(N, num_partitions, g, h)
        return s, w

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            task = self.task_queue.get()
            if isinstance(task, EndTask):
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

    

