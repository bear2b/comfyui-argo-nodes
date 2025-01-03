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
            print("adding alpha channel")
            alpha = torch.ones((image_tensor.shape[0], 1, image_tensor.shape[2], image_tensor.shape[3]), 
                               device=image_tensor.device)
            image_tensor = torch.cat((image_tensor, alpha), dim=1)
        
        # Reshape to (B, H*W, C) where C is 4
        B, C, H, W = image_tensor.shape
        #print(B, C, H, W)
        image_flat = image_tensor.permute(0, 2, 3, 1).reshape(-1, C)
        #image_flat = torch.reshape(image_tensor.permute(0, 2, 3, 1), (-1, 4))
        
        # Apply color matrix
        color_matrix = torch.tensor(color_matrix, device=image_tensor.device, dtype=image_tensor.dtype)
        transformed = torch.matmul(image_flat, color_matrix.T)
        
        # Reshape back to (B, C, H, W)
        image_transformed = transformed.view(B, H, W, C).permute(0, 3, 1, 2)
        #B, C, H, W = image_transformed.shape
        #print(B, C, H, W)
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
        
        if isinstance(image, Image.Image):
            print("image instance")
            image_np = np.array(image).astype(np.float32) #/ 255.0  # Normalize
        elif isinstance(image, np.ndarray):
            print("np.ndarray instance")
            image_np = image.astype(np.float32) #/ 255.0
        elif torch.is_tensor(image):
            print("tensor instance")
            image_np = image.cpu().numpy().astype(np.float32) #/ 255.0
        else:
            raise ValueError("Unsupported image type. Expected PIL Image, NumPy array, or PyTorch tensor.")
        
        
        # Add batch dimension if missing
        if image_np.ndim == 3:  # (H, W, C)
            print("expanding dim ==3")
            image_np = np.expand_dims(image_np, axis=0)  # (1, H, W, C)
        
        # Convert NumPy array to Tensor
        image_tensor = torch.from_numpy(image_np).permute(0, 3, 1, 2).to('cuda')
        
        # Apply color matrix
        result_tensor = self.apply_color_matrix(image_tensor, matrix)
        
        # Convert back to PIL image
        B, C, H, W = result_tensor.shape
        print("result tensor", B, C, H, W)
        #result_np = (result_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        #result_np = (result_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        resfinal = result_tensor.permute(1, 0, 2, 3) #result_tensor[:, :3, :, :].permute(1, 0, 2, 3)
        #resfinal = result_tensor[:, :3, :, :].squeeze(0).permute(1, 2, 0).unsqueeze(0)
        B, C, H, W = resfinal.shape
        print("resfinal", B, C, H, W)
        #return (resfinal[0],)
        return resfinal

        #if C==4 :
        #    return (Image.fromarray(result_np, mode="RGBA"),)
        #else :
        #    return (Image.fromarray(result_np, mode="RGB"),)


NODE_CLASS_MAPPINGS = {
    'ColorMatrixGPU': ColorMatrixGPUNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ColorMatrixGPU': 'Color Matrix (GPU)'
}
