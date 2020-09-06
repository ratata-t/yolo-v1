from yolo. dataset import load_dataset


def get_obj_names():
    voc_train = load_dataset()
    set_name = set()
    for row in voc_train:
        l, r = row
        objects = r['annotation']['object']
        for obj in objects:
            set_name.add(obj['name'])

    return set_name
