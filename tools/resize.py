from PIL import Image

def resize_image(input_path, output_path, size):
    """
    Resize an image to the given size and save it.

    Parameters:
        input_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        size (tuple): New size as (width, height).
    """
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        resized_img.save(output_path)
        print(f"Image saved to {output_path} with size {size}")

# Example usage
if __name__ == "__main__":
    input_image_path = "/home/zhiyuhan/Downloads/1000054625.jpg"           # Replace with your input image path
    output_image_path = "resized_output.jpg" # Replace with desired output path
    new_size = (960, 1280)                    # Replace with desired (width, height)
    
    resize_image(input_image_path, output_image_path, new_size)
