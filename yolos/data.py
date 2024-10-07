import torch
from torchvision.datasets.voc import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

image_size = (448, 448) # size used in the paper
transform = transforms.Compose([
  transforms.Resize(image_size),
  transforms.ToTensor(),
])

# pascal voc dataset
dataset = VOCDetection(
  root="data",
  year="2007",
  image_set="trainval",
  download=True,
  transform=transform,
)

class DetectionDataset(Dataset):
  def __init__(self, dataset, n_grid, n_box, n_class):
    self.dataset = dataset
    self.n_grid = n_grid
    self.n_box = n_box
    self.n_class = n_class
    self.classes_idx = {}

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    input, label = self.dataset[idx]
    c, w, h = input.shape # (channels, width, height)
    grid_size_x = w / self.n_grid
    grid_size_y = h / self.n_grid
    target = torch.zeros((self.n_grid, self.n_grid, self.n_box * 5 + self.n_class))

    for obj in label["annotation"]["object"]:
      name, bbox = obj["name"], obj["bndbox"]
      if name not in self.classes_idx:
          self.classes_idx[name] = len(self.classes_idx)
      indice = self.classes_idx[name]

      x_min, y_min, x_max, y_max = map(int, (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
      mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2
      col, row = int(mid_x // grid_size_x), int(mid_y // grid_size_y)

      if row >= self.n_grid or col >= self.n_grid:
        continue
      
      # assign bounding box
      for box_index in range(self.n_box):
        if target[row, col, box_index * 5] == 0: 
          bbox_truth = torch.tensor([
            (mid_x - col * grid_size_x) / grid_size_x,
            (mid_y - row * grid_size_y) / grid_size_x,
            (x_max - x_min) / w,
            (y_max - y_min) / h,
          ])
          target[row, col, box_index * 5:box_index * 5 + 4] = bbox_truth
          target[row, col, box_index * 5 + 4] = 1.0  # confidence score 
          target[row, col, self.n_box * 5 + indice] = 1.0  # class one-hot encoding 
          break

    return input, target
  
def get_data_loader(dataset, train=True, batch_size=32, shuffle=False):
  pass
