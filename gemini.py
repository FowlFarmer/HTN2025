import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from google import genai
from google.genai import types

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set API_KEY in your environment")

client = genai.Client(api_key=API_KEY)

def edit_with_image(image_path: str, text_prompt: str,
                    model: str = "gemini-2.5-flash-image-preview"):
    # Load input image (PIL is fine; bytes also work)
    pil_in = Image.open(image_path)

    # Ask for IMAGE output; send both text + image as contents
    resp = client.models.generate_content(
        model=model,
        contents=[text_prompt, pil_in],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]  # ensures image(s) in response
        ),
    )

    # Collect any image parts returned
    out_imgs = []
    for cand in resp.candidates or []:
        for part in cand.content.parts or []:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                # bytes -> NumPy (OpenCV BGR)
                img_bytes = part.inline_data.data
                arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                out_imgs.append(img)

    if not out_imgs:
        raise RuntimeError("No image returned by the model.")
    return out_imgs

if __name__ == "__main__":
    images = edit_with_image(
        "/Users/theodore/Desktop/Repositories/HTN2025/pics/burning_monk.png",
        "Make this look like a pencil sketch with heavier outlines"
    )
    cv2.imshow("Gemini Output", images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output.png", images[0])