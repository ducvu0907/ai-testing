import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class YOLOV1(nn.Module):
  def __init__(self, n_grid, n_box, n_class):
    super(YOLOV1, self).__init__()
    self.n_grid = n_grid
    self.n_box = n_box
    self.n_class = n_class

    self.layers = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
      nn.LeakyReLU(negative_slope=0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(64, 192, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(192, 128, kernel_size=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(256, 256, kernel_size=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(256, 512, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(512, 256, kernel_size=1),
      nn.Conv2d(256, 512, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),

      *[nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=1),
          nn.Conv2d(256, 512, kernel_size=3, padding=1),
          nn.LeakyReLU(negative_slope=0.1)
      ) for _ in range(4)],

      nn.Conv2d(512, 512, kernel_size=1),
      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(1024, 512, kernel_size=1),
      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),

      *[nn.Sequential(
          nn.Conv2d(1024, 512, kernel_size=1),
          nn.Conv2d(512, 1024, kernel_size=3, padding=1),
          nn.LeakyReLU(negative_slope=0.1)
      ) for _ in range(2)],

      nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
      nn.LeakyReLU(negative_slope=0.1),

      nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
      nn.LeakyReLU(negative_slope=0.1),

      nn.Flatten(), # flattent conv map to n_grid * n_grid * 1024 feature vector
      nn.Linear(n_grid * n_grid * 1024, 4096),
      nn.Dropout(),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Linear(4096, n_grid * n_grid * (n_box * 5 + n_class)),
    )

  def forward(self, x):
    # x: (batch_size, channels, width, height)
    batch_size = x.shape[0]
    x = self.layers(x)
    x = x.reshape(batch_size, self.n_grid, self.n_grid, self.n_box * 5 + self.n_class) # (batch_size, S, S, B*5+C)
    return x

if __name__ == "__main__":
  pass
