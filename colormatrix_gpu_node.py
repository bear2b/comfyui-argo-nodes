import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import json
import os

def load_aws_credentials(file_path):
    """Load AWS credentials from a JSON file."""
    with open(file_path, 'r') as f:
        credentials = json.load(f)
    return credentials

# Load credentials from file
base_dir = os.path.dirname(__file__)  # Directory of the current script
credentials_file = os.path.join(base_dir, "aws_credentials.json")
credentials = load_aws_credentials(credentials_file)


# Initialize the S3 client with the loaded credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id=  os.getenv("S3_ACCESS_KEY") if os.getenv("S3_ACCESS_KEY") else credentials["aws_access_key_id"],
    aws_secret_access_key=os.getenv("S3_SECRET_KEY") if os.getenv("S3_SECRET_KEY") else credentials["aws_secret_access_key"],
    region_name=os.getenv("S3_REGION") if os.getenv("S3_REGION") else credentials["region_name"]
)

bucket_name = os.getenv("S3_BUCKET_NAME") if os.getenv("S3_BUCKET_NAME") else credentials["bucket_name"]


class ColorMatrixGPUNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_matrix_4x4_csv": ("STRING",),  # Expect a 4x4 color matrix in string format (comma-separated)
                "add_vec4_csv": ("STRING",)  # Expect a 4x4 color matrix in string format (comma-separated)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Image/Processing"

    def apply_color_matrix(self, image_tensor, color_matrix, add_vector):
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
        add_vector = torch.tensor(add_vector, device=image_tensor.device, dtype=image_tensor.dtype)
        transformed = torch.matmul(image_flat, color_matrix.T) + add_vector
        
        # Reshape back to (B, C, H, W)
        image_transformed = transformed.view(B, H, W, C).permute(0, 3, 1, 2)
        #B, C, H, W = image_transformed.shape
        #print(B, C, H, W)
        return image_transformed.clamp(0, 1)

    def run(self, image, color_matrix_4x4_csv, add_vec4_csv):
        # Parse color matrix string
        try:
            matrix_values = [float(x) for x in color_matrix_4x4_csv.split(',')]
            if len(matrix_values) != 16:
                raise ValueError("Color matrix must have 16 values.")
            matrix = np.array(matrix_values, dtype=np.float32).reshape(4, 4)
        except Exception as e:
            raise ValueError(f"Invalid color matrix: {e}")
        
        try:
            vector_values = [float(x) for x in add_vec4_csv.split(',')]
            if len(vector_values) != 4:
                raise ValueError("Vector must have 4 values.")
            vector = np.array(vector_values, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Invalid color add vector: {e}")
        
		
        if isinstance(image, Image.Image):
            print("image instance")
            image_np = np.array(image).astype(np.float32) / 255.0  # Normalize
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
        result_tensor = self.apply_color_matrix(image_tensor, matrix, vector)
        
        # Convert back to PIL image
        B, C, H, W = result_tensor.shape
        resfinal = result_tensor.permute(0, 2, 3, 1)
        return (resfinal,)




def fetch_image(index, url):
    """Fetch an image from a URL and return it as a PIL.Image object."""
    response = requests.get(url)
    response.raise_for_status()
    return index, Image.open(BytesIO(response.content)).convert("RGB")

def create_image_grid(image_urls, grid_size=(5, 5), cell_size=(256, 256)):
    
    if len(image_urls) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Expected {grid_size[0] * grid_size[1]} image URLs, got {len(image_urls)}")

    # Fetch and resize images
    #images = [resize_image(fetch_image(url), cell_size) for url in image_urls]
    #images = [fetch_image(url) for url in image_urls]
    images = [None] * len(image_urls)
    with ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(fetch_image, idx, url): idx
            for idx, url in enumerate(image_urls)
        }
        for future in as_completed(future_to_index):
            try:
                index, image = future.result()
                images[index] = image  # Assign to the correct position
            except Exception as e:
                url = image_urls[future_to_index[future]]
                print(f"Error fetching {url}: {e}")

    if any(img is None for img in images):
        raise RuntimeError("Some images failed to load.")

    # Create the grid canvas
    grid_width = grid_size[1] * cell_size[0]
    grid_height = grid_size[0] * cell_size[1]
    grid_image = Image.new("RGB", (grid_width, grid_height))

    # Paste images into the grid
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        x = col * cell_size[0]
        y = row * cell_size[1]
        grid_image.paste(img, (x, y))

    return grid_image


class LoadGridFromURL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefix": ("STRING",),
                "names": ("STRING",),  # Expect a 5x5 string format csv
                "suffix": ("STRING",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Image/Processing"

    def run(self, prefix, names, suffix):
        # names = "3_21_0,4_21_0,5_21_0,6_21_0,7_21_0,3_22_0,4_22_0,5_22_0,6_22_0,7_22_0,3_23_0,4_23_0,5_23_0,6_23_0,7_23_0,3_24_0,4_24_0,5_24_0,6_24_0,7_24_0,3_25_0,4_25_0,5_25_0,6_25_0,7_25_0"
        # names = "3_21,4_21,5_21,6_21,7_21,3_22,4_22,5_22,6_22,7_22,3_23,4_23,5_23,6_23,7_23,3_24,4_24,5_24,6_24,7_24,3_25,4_25,5_25,6_25,7_25"
        try:
            images = [(prefix+x+suffix) for x in names.split(',')]
            if len(images) != 25:
                raise ValueError("Image must have 25 values.")
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")

        #images = [(suffix+url+prefix) for url in names]
        #print(images)
        image_np = np.array(create_image_grid(images)).astype(np.float32) / 255.0
        return (torch.from_numpy(image_np).unsqueeze(0),)



def split_image_to_grid(input_image, grid_size=(5, 5)):
    width, height = input_image.size
    cell_width = width // grid_size[1]
    cell_height = height // grid_size[0]

    grid_images = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            grid_images.append(input_image.crop((left, upper, right, lower)))

    return grid_images

def upload_image_to_s3(index, image, bucket_name, object_name, s3_client):
    # Convert image to a bytes buffer
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Upload the image to S3
    s3_client.upload_fileobj(buffer, bucket_name, object_name)
    return index, True


class SaveGridToS3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prefix": ("STRING",),
                "names": ("STRING",),  # Expect a 5x5 string format csv
                "suffix": ("STRING",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("s3_image_paths",)
    FUNCTION = "run"
    CATEGORY = "output"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    def run(self, image, prefix, names, suffix):
        # names = "3_21_1,4_21_1,5_21_1,6_21_1,7_21_1,3_22_1,4_22_1,5_22_1,6_22_1,7_22_1,3_23_1,4_23_1,5_23_1,6_23_1,7_23_1,3_24_1,4_24_1,5_24_1,6_24_1,7_24_1,3_25_1,4_25_1,5_25_1,6_25_1,7_25_1"
        try:
            returnValue = [(x+"") for x in names.split(',')]
            images = [(prefix+x+suffix) for x in names.split(',')]
            if len(images) != 25:
                raise ValueError("Image must have 25 values.")
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")

        #for image in images:
        imag = image[0] #only send one image
        i = 255. * imag.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        try:
            # Split the image into a grid
            grid_images = split_image_to_grid(img)

            # Upload each grid image to S3
            #for idx, image in enumerate(grid_images):
            #    object_name = f"{images[idx]}"
            #    upload_image_to_s3(image, bucket_name, object_name, s3_client)
            bools = [None] * len(grid_images)
            with ThreadPoolExecutor() as executor:
                future_to_index = {
                    executor.submit(upload_image_to_s3, idx, image, bucket_name, f"{images[idx]}", s3_client): idx
                    for idx, image in enumerate(grid_images)
                }
                for future in as_completed(future_to_index):
                    try:
                        index, success = future.result()
                        bools[index] = success
                    except Exception as e:
                        print(f"Error posting : {e}")
            if any(img is None for img in bools):
                raise RuntimeError("Some images failed to upload.")
        except Exception as e:
            raise ValueError(f"Invalid input: {e}")
        
        return { "ui": { "images": (returnValue,) }, "result": (returnValue,) } # "ui": { "images": results },
        



NODE_CLASS_MAPPINGS = {
    'ColorMatrixGPU': ColorMatrixGPUNode,
    'LoadGridFromURL': LoadGridFromURL,
    'SaveGridToS3': SaveGridToS3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ColorMatrixGPU': 'Color Matrix (GPU)',
    'LoadGridFromURL': 'Load Grid From URL',
    'SaveGridToS3': 'Save Grid To S3'
}
