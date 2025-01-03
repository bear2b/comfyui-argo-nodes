import torch
import numpy as np
from PIL import Image
from comfy.model_base import Node

class ColorMatrixGPUNode(Node):
    def __init__(self):
        super().__init__()
        self.input_types = {
            'image': ('IMAGE',),
            'color_matrix': ('STRING',)  # Expect a 4x4 color matrix in string format (comma-separated)
        }
        self.output_types = {'image': 'IMAGE'}

    def apply_color_matrix(self, image_tensor, color_matrix):
        # Ensure image is in the correct shape (C, H, W)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        
        # Add alpha channel if missing
        if image_tensor.shape[0] == 3:
            alpha = torch.ones((1, image_tensor.shape[1], image_tensor.shape[2]), device=image_tensor.device)
            image_tensor = torch.cat((image_tensor, alpha), dim=0)
        
        # Reshape to (H*W, 4)
        H, W = image_tensor.shape[1], image_tensor.shape[2]
        image_flat = image_tensor.view(4, -1).T
        
        # Apply color matrix
        color_matrix = torch.tensor(color_matrix, device=image_tensor.device, dtype=image_tensor.dtype)
        transformed = torch.matmul(image_flat, color_matrix.T)
        
        # Reshape back to (4, H, W)
        image_transformed = transformed.T.view(4, H, W)
        return image_transformed.clamp(0, 1)

    def run(self, image, color_matrix):
        # Convert input color matrix string to a tensor
        try:
            matrix_values = [float(x) for x in color_matrix.split(',')]
            if len(matrix_values) != 16:
                raise ValueError("Color matrix must have 16 values.")
            matrix = np.array(matrix_values, dtype=np.float32).reshape(4, 4)
        except Exception as e:
            raise ValueError(f"Invalid color matrix: {e}")
        
        # Ensure image is a tensor on GPU
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to('cuda')
        
        # Apply color matrix
        result_tensor = self.apply_color_matrix(image_tensor, matrix)
        
        # Convert back to PIL image
        result_image = (result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(result_image)

NODE_CLASS_MAPPINGS = {
    'ColorMatrixGPU': ColorMatrixGPUNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ColorMatrixGPU': 'Color Matrix (GPU)'
}
