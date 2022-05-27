import numpy as np
from forte.data.ontology.top import BoundingBox, Link, Annotation
from forte.data.data_pack import DataPack

datapack = DataPack("image")

# line = np.zeros((6, 12))
line = np.zeros((20, 20))
line[2, 2] = 1
line[3, 3] = 1
line[4, 4] = 1
datapack.payloads.append(line)
datapack.payloads.append(line)
# grid config: 3 x 4
# grid cell indices: (0, 0)
bb1 = BoundingBox(datapack, 0, 2, 2, 3, 4, 0, 0)
datapack.image_annotations.append(bb1)
# grid config: 3 x 4
# grid cell indices: (1, 0)
bb2 = BoundingBox(datapack, 0, 2, 2, 3, 4, 1, 0)
datapack.image_annotations.append(bb2)
# grid config: 4 x 4
# grid cell indices: (1, 0)
bb3 = BoundingBox(datapack, 0, 2, 2, 4, 4, 0, 0)

print(bb1.is_overlapped(bb2))
print(bb1.is_overlapped(bb3))

datapack.set_text("bb1, bb2, bb3")
bb1_descrip = Annotation(datapack, 0, 3)

print(bb1_descrip.text)
link1 = Link(datapack, bb1_descrip, bb1)
datapack.add_entry(link1)
print(list(datapack.all_links))

print(bb1.compute_iou(bb3))
