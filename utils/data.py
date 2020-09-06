class Bbox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @property
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
