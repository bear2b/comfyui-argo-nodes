import torch
import numpy as np
from PIL import Image

class ColorMatrixGPUNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_matrix": ("STRING",)  # Expect a 4x4 color matrix in string format (comma-separated)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Image/Processing"

    def apply_color_matrix(self, image_tensor, color_matrix):
        # Ensure image is in the correct shape (B, C, H, W)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Add alpha channel if missing
        if image_tensor.shape[1] == 3:
            alpha = torch.ones((image_tensor.shape[0], 1, image_tensor.shape[2], image_tensor.shape[3]), device=image_tensor.device)
            image_tensor = torch.cat((image_tensor, alpha), dim=1)
        
        # Reshape to (B, H*W, C) where C is 4
        B, C, H, W = image_tensor.shape
        image_flat = image_tensor.permute(0, 2, 3, 1).reshape(-1, C)  # Flatten (B, H, W, 4) to (B*H*W, 4)
        
        # Apply color matrix
        color_matrix = torch.tensor(color_matrix, device=image_tensor.device, dtype=image_tensor.dtype)
        transformed = torch.matmul(image_flat, color_matrix.T)
        
        # Reshape back to (B, H, W, 4)
        image_transformed = transformed.view(B, H, W, 4).permute(0, 3, 1, 2)
        return image_transformed.clamp(0, 1)


    def run(self, image, color_matrix):
        # Parse color matrix string
        try:
            matrix_values = [float(x) for x in color_matrix.split(',')]
            if len(matrix_values) != 16:
                raise ValueError("Color matrix must have 16 values.")
            matrix = np.array(matrix_values, dtype=np.float32).reshape(4, 4)
        except Exception as e:
            raise ValueError(f"Invalid color matrix: {e}")
        
        # Convert image to tensor (B, H, W, C) to (B, C, H, W)
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to('cuda')
        
        # Apply color matrix
        result_tensor = self.apply_color_matrix(image_tensor, matrix)
        
        # Convert back to PIL image
        result_image = (result_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        return (Image.fromarray(result_image),)

NODE_CLASS_MAPPINGS = {
    'ColorMatrixGPU': ColorMatrixGPUNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ColorMatrixGPU': 'Color Matrix (GPU)'
}
