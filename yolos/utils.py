def compute_loss(preds, targets):
  # preds, targets: (batch_size, grid_size, grid_size, num_box * 5 + num_class)
  l_coord, l_noobj = 5, 0.5


def get_iou(pbox, tbox):
  # pbox, tbox = (x_min, y_min, width, height)
  # iou = area_overlap / area_union
  pbox, tbox = get_box_coords(pbox), get_box_coords(tbox)
  x1_min, x1_max, y1_min, y1_max = pbox
  x2_min, x2_max, y2_min, y2_max = tbox
  area_pbox = (x1_max - x1_min) * (y1_max - y1_min)
  area_tbox = (x1_max - x1_min) * (y1_max - y1_min)


def get_box_coords(box):
  # box: (x_min, y_min, width, height)
  x_min, y_min, width, height = box
  x_max = x_min + width
  y_max = y_min + height
  return (x_min, x_max, y_min, y_max)

def train(model):
  pass


def plot_boxes():
  pass