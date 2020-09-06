from yolo.dataset import load_dataset
import config


def get_obj_names():
    voc_train = load_dataset()
    set_name = set()
    for row in voc_train:
        l, r = row
        objects = r['annotation']['object']
        for obj in objects:
            set_name.add(obj['name'])

    return set_name


class Bbox:
    s_width = config.WIDTH / config.S
    s_height = config.HEIGHT / config.S

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax        

    @property
    def x_c(self):
        return (self.xmin + self.xmax) / 2.0
    
    @property
    def y_c(self):
        return (self.ymin + self.ymax) / 2.0

    @property
    def section_x(self):
        return int(self.x_c / self.s_width)

    @property
    def section_y(self):
        return int(self.y_c / self.s_height)

    @property
    def x_gt(self):
        return (self.x_c - self.s_width * (self.section_x + 0.5)) / self.s_width

    @property
    def y_gt(self):
        return (self.y_c - self.s_height * (self.section_y + 0.5)) / self.s_height

    @property
    def w_gt(self):
        return (self.xmax - self.xmin)/self.s_width # gt - groung truth

    @property
    def h_gt(self):
        return (self.ymax - self.ymin)/self.s_height
    
    @property
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)
