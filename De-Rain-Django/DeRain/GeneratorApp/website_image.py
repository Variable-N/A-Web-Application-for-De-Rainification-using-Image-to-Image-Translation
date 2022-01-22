from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from .generator_model import Generator
from torchvision.utils import save_image
from skimage import io



def main():

    DEVICE = "cpu"
    def save_some_example(gen, val_loader, folder):
        x = next(iter(val_loader))
        x = x.to(DEVICE)
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5
            save_image(y_fake, folder + f"/y_gen.jpg")
        gen.train()

    transform_only_input = A.Compose(
            [
                A.Resize(width=512, height=512), 
                A.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value = 255.0,),
                ToTensorV2()
            ]
        )

    class WebsiteDataset(Dataset):
        def __init__(self,root_dir):
            self.root_dir = root_dir
            self.list_files = os.listdir(self.root_dir)
            
        def __len__(self):
            return len(self.list_files)

        def __getitem__(self, index):
            img_file = self.list_files[index]
            img_path = os.path.join(self.root_dir,img_file)
            image = np.array(Image.open(img_path))
            input_image = transform_only_input(image=image)["image"]
            return input_image




    train_dataset = WebsiteDataset(root_dir="media/my_image")
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


    model_B = Generator().to(DEVICE)
    model_B.load_state_dict(torch.load('GeneratorApp/gen_95.pth', map_location=torch.device('cpu')))
    with torch.no_grad():
        save_some_example(model_B, test_loader, "media/my_image")