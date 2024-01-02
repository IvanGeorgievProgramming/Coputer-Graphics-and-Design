from PIL import Image

# * Function to load image from file
def open_image(filename):
    image = Image.open(filename)
    new_image = list(image.getdata())
    width, height = image.size
    return [new_image[i * width:(i + 1) * width] for i in range(height)]

# * Function to trace the outline of the image
def trace(image, start_y, start_x):
    height, width = len(image), len(image[0])
    new_image = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    orientation = 0
    y, x = start_y, start_x
    first_pass = True
    traced_twice = False

    while True:
        new_image[y][x] = (0, 0, 0)

        if y == start_y and x == start_x and not first_pass:
            if traced_twice:
                break
            traced_twice = True

        dy, dx = directions[orientation]
        next_y, next_x = y + dy, x + dx

        if 0 <= next_y < height and 0 <= next_x < width and image[next_y][next_x] != (255, 255, 255):
            y, x = next_y, next_x
            first_pass = False
            orientation = (orientation + 3) % 4
        else:
            orientation = (orientation + 1) % 4

    return new_image

# * Function to find the first black pixel in the image and trace its outline using trace()
def trace_outline_from_black_pixel(image):
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            if pixel == (0, 0, 0):
                if y == 0 or image[y-1][x] != (0, 0, 0):
                    return trace(image, y, x)

# * Function to save image to file
def save_image(image, filename):
    height, width = len(image), len(image[0])
    image_pil = Image.new("RGB", (width, height))
    for i in range(height):
        for j in range(width):
            image_pil.putpixel((j, i), image[i][j])
    image_pil.save(filename)

# * Main function
def main():
    input_filename = "original_image.png"
    output_filename = "new_image.png"

    image = open_image(input_filename)
    traced_image = trace_outline_from_black_pixel(image)

    save_image(traced_image, output_filename)

# * Execute the main function
if __name__ == "__main__":
    main()