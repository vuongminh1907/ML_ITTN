import numpy as np
from PIL import Image
from src.core import process_inpaint


def remove_person(background, mask):
    # Load background and mask images
    background = background.convert("RGBA")
    mask = mask.convert("RGBA")

    #resize anh co the bo qua 
    background = background.resize((768, 1024))
    mask = mask.resize((768, 1024))

    # Convert images to numpy arrays
    background = np.array(background)
    mask = np.array(mask)

    background_img = np.where(
            (mask[:, :, 0] == 0) & 
            (mask[:, :, 1] == 0) & 
            (mask[:, :, 2] == 0)
        )
    drawing = np.where(
        (mask[:, :, 0] == 255) & 
        (mask[:, :, 1] == 255) & 
        (mask[:, :, 2] == 255)
    )
    mask[background_img]=[0,0,0,255]
    mask[drawing]=[0,0,0,0] # RGBA

    # Process the image
    output = process_inpaint(background, mask)
    img_output = Image.fromarray(output).convert("RGB")
    
    return img_output  

if __name__ == "__main__":
    background_path = "c2.jpeg"       # Đường dẫn tới ảnh nền
    mask_path = "black_mask.jpg"      # Đường dẫn tới ảnh mask

    background = Image.open(background_path)
    mask = Image.open(mask_path)

    # Save the result
    img_output = remove_person(background, mask)
    img_output.save("output.jpg")