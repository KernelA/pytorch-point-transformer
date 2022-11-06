import enum


class ModelNetType(enum.IntEnum):
    modelnet_10 = 10
    modelnet_40 = 40


MODELNET_CLASSES = {
    ModelNetType.modelnet_10: ["bathtub", "bed", "chair", "desk", "dresser",
                               "monitor", "night_stand", "sofa", "table", "toilet"],
    ModelNetType.modelnet_40: ["airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
                               "chair", "cone", "cup", "curtain",
                               "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard",
                               "lamp", "laptop", "mantel", "monitor", "night_stand", "person", "piano", "plant",
                               "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table", "tent", "toilet",
                               "tv_stand", "vase", "wardrobe", "xbox"]
}
