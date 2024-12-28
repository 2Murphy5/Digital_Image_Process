import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import time

class MedicalImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Segmentation System")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(r"E:\\pycharm_pro\\Unet\\MyUnet\\preprocessing\\1228model.pth")
        self.image_path = None
        self.mask = None

        # UI Elements
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Processing", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.done_button = tk.Button(root, text="Done", command=self.finish_app, state=tk.DISABLED)
        self.done_button.pack(pady=10)

        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10)

        self.time_label = tk.Label(root, text="", font=("Arial", 12))
        self.time_label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def load_model(self, model_path):
        model = deeplabv3_resnet50(pretrained=False, num_classes=1)  # Set num_classes to 1 for binary segmentation
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.process_button.config(state=tk.NORMAL)

    def process_image(self):
        if not self.image_path:
            return

        start_time = time.time()
        self.progress.start(10)

        # Predict mask
        self.mask = self.predict_mask(self.image_path)

        self.progress.stop()
        elapsed_time = time.time() - start_time
        self.time_label.config(text=f"Processing Time: {elapsed_time:.2f} seconds")

        self.display_image(self.image_path, self.mask)
        self.done_button.config(state=tk.NORMAL)

    def predict_mask(self, image_path):
        transform = transforms.Compose([transforms.ToTensor()])
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out']
            output = torch.sigmoid(output).squeeze(0).cpu().numpy()

        mask = (output > 0.5).astype(np.uint8) * 255  # Binary segmentation assumes single output channel
        return mask

    def display_image(self, image_path, mask=None):
        image = Image.open(image_path).convert("RGB")

        if mask is not None:
            # Ensure mask is in 2D uint8 format
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]  # Remove channel dimension if present
            elif mask.ndim > 2:
                mask = mask.squeeze()  # Handle unexpected dimensions
            mask = mask.astype(np.uint8)  # Ensure uint8 type
            mask_image = Image.fromarray(mask, mode="L")  # Convert to grayscale PIL Image
            image.paste(mask_image, (0, 0), mask_image)

        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)

        self.image_label.config(image=photo)
        self.image_label.image = photo

        # Enable click-to-download if mask is available
        if mask is not None:
            self.image_label.bind("<Button-1>", lambda e: self.download_mask(mask))

    def download_mask(self, mask):
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]  # Remove channel dimension if present
            elif mask.ndim > 2:
                mask = mask.squeeze()  # Handle unexpected dimensions
            mask = mask.astype(np.uint8)  # Ensure uint8 type
            mask_image = Image.fromarray(mask, mode="L")  # Convert to grayscale PIL Image
            mask_image.save(save_path)
            print(f"Mask saved to {save_path}")

    def finish_app(self):
        messagebox.showinfo("Get Well Soon", "Get Well Soon! The application will now close.")
        self.root.after(3000, self.exit_app)

    def exit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalImageSegmentationApp(root)
    root.mainloop()
