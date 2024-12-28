import os
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm


class BatchPrediction:
    def __init__(self, model_path, device):
        self.device = device
        self.model = deeplabv3_resnet50(pretrained=False,
                                        num_classes=2)  # Ensure num_classes matches your trained model
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out']
            output = torch.sigmoid(output).squeeze(0).cpu().numpy()

        output_mask = (output[1] > 0.5).astype(np.uint8) * 255  # Assuming binary segmentation
        return output_mask

    def batch_predict(self, image_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_dir, image_file)
            output_mask = self.predict(image_path)

            mask_filename = os.path.splitext(image_file)[0] + ".png"
            output_path = os.path.join(output_dir, mask_filename)

            mask_image = Image.fromarray(output_mask)
            mask_image.save(output_path)
            print(f"Saved mask for {image_file} to {output_path}")


if __name__ == "__main__":
    model_path = "/home/yuanweimin/DeepLabV3Plus/models/1228_model.pth"  # Path to the trained model
    input_images_dir = "/home/yuanweimin/DFUC_Data_Test/images"  # Directory containing input images
    output_masks_dir = "/home/yuanweimin/DeepLabV3Plus/1228TestMasks"  # Directory to save output masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = BatchPrediction(model_path, device)
    predictor.batch_predict(input_images_dir, output_masks_dir)
