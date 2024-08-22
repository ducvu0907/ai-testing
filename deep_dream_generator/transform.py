import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def _transform(model, layer_index, image_path, iterations, learning_rate):
  image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # scale to ImageNet
  ])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  image = Image.open(image_path).convert("RGB")
  image = image_transforms(image)
  image = image.unsqueeze(0).to(device) # add batch dimension
  image.requires_grad = True
  model = model.to(device).eval()

  for iter in range(iterations):
    model.zero_grad()
    out = image
    for idx, layer in model.features.named_children():
      out = layer(out)
      if idx == layer_index:
        break
    loss = out.norm()
    loss.backward()
    with torch.no_grad():
      image += learning_rate * image.grad
      image.grad.zero_()

    return image

def deepdream(image_path):
  model = models.vgg16()
  layer_index = 28
  iterations = 100
  learning_rate = 0.01
  try:
    image = _transform(model, layer_index, image_path, iterations, learning_rate)
    image = image.squeeze(0).cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0)) # convert to (H, W, C)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)

    # save
    image = Image.fromarray((image * 255).astype(np.uint8))
    result_filename = f"transformed_{image_path.split('/')[-1]}"
    print(result_filename)
    image.save(f"uploads/{result_filename}")
    return result_filename

  except Exception as e:
    print(f"An error occured: {e}")

# testing
if __name__ == "__main__":
  pass