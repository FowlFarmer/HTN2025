import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Image preprocessing pipeline for painting vectorization.

    Args:
        image_path (str): Path to the input image

    Returns:
        tuple: (processed_image, scale_factor, original_dimensions)
            processed_image: numpy.ndarray with shape (height, width, 3),
                           dtype float32, values in range [0.0, 1.0], RGB color order
            scale_factor: float, resize ratio applied to original image
            original_dimensions: tuple (original_height, original_width) in pixels
    """

    # 1.1 Image Loading and Format Conversion
    img = cv2.imread(image_path)[:, :, ::-1]  # BGR -> RGB
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Handle RGBA to RGB conversion
    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]  # Remove alpha channel

    # Store original dimensions
    original_h, original_w = img.shape[:2]

    # 1.2 Resizing Strategy
    h, w = img.shape[:2]
    scale = 1024 / max(h, w) if max(h, w) > 1024 else 1.0
    img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 1.3 Denoising (Optional) - Skip for now unless needed

    # 1.4 Contrast Enhancement
    lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img_small = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)

    # 1.5 Data Type Conversion
    img_processed = img_small.astype(np.float32) / 255.0

    return img_processed, scale, (original_h, original_w)

if __name__ == "__main__":
    # Process the monet.jpeg image
    processed_img, scale_factor, original_dims = preprocess_image("monet.jpeg")

    print(f"Original dimensions: {original_dims}")
    print(f"Scale factor: {scale_factor}")
    print(f"Processed image shape: {processed_img.shape}")
    print(f"Processed image dtype: {processed_img.dtype}")
    print(f"Processed image range: [{processed_img.min():.3f}, {processed_img.max():.3f}]")