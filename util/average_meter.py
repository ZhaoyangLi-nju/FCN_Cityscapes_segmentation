# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = round(self.sum / self.count, 3)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1.0,acc_Flag=False):
        if not self.initialized:
            self.initialize(val, weight)
        elif not acc_Flag:
            self.add(val, weight)
        else:
            self.add1(val, weight)

    def add1(self, val, weight):
        self.val = val
        self.sum += val * float(weight)
        self.count += weight
        self.avg = float(self.sum) / float(self.count)
    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum/ float(self.count)

    def value(self):
        return self.val

    def average(self):
        return self.avg