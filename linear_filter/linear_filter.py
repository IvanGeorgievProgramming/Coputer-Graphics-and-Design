import cv2
import numpy as np
from matplotlib import pyplot as plt

# * Function to apply the convolution
def apply_kernel_filter(image, kernel):
    # Flip the kernel for convolution
    kernel = np.flipud(np.fliplr(kernel))
    # Apply the filter using OpenCV's filter2D function for better performance
    return cv2.filter2D(image, -1, kernel)

# * Function to plot the filtered image
def plot_filtered_image(filtered_image, title):
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# * Function to display the original filter
def display_original_filter(img):
    # Define the Original Kernel
    original_kernel = np.array([[0,  0,  0,  0,  0],
                                [0,  0,  0,  0,  0],
                                [0,  0,  1,  0,  0],
                                [0,  0,  0,  0,  0],
                                [0,  0,  0,  0,  0]], np.float32)
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, original_kernel)

    # Clip the values to [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255).astype('uint8')

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Original Filter')

# * Function to display the average filter
def display_average_filter(img):
    # Define the Average Kernel
    average_kernel  = np.array([[1,  1,  1,  1,  1],
                                [1,  1,  1,  1,  1],
                                [1,  1,  1,  1,  1],
                                [1,  1,  1,  1,  1],
                                [1,  1,  1,  1,  1]], np.float32)
    # Normalize the kernel (make the sum of all elements equal to 1)
    average_kernel /= np.sum(average_kernel)
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, average_kernel)

    # Clip the values to [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255).astype('uint8')

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Average Filter')

# * Function to display the gaussian blur
def display_gaussian_blur(img):
    # Define the kernel size and sigma value
    kernel_size = 5
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # Create a grid of (x,y) coordinates
    x = np.arange(0, kernel_size) - (kernel_size - 1) / 2.0
    y = np.arange(0, kernel_size) - (kernel_size - 1) / 2.0
    xx, yy = np.meshgrid(x, y)

    # Calculate the Gaussian function
    gaussian_kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    # Normalize the kernel (make the sum of all elements equal to 1)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, gaussian_kernel)

    # Clip the values to [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255).astype('uint8')

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Gaussian Blur')

# * Function to display the sharpen filter
def display_sharpen_filter(img):
    # Define the Sharpen Kernel
    sharpen_kernel  = np.array([[0,  0,  0,  0,  0],
                                [0,  0, -1,  0,  0],
                                [0, -1,  5, -1,  0],
                                [0,  0, -1,  0,  0],
                                [0,  0,  0,  0,  0]], np.float32)
    # Normalize the kernel (make the sum of all elements equal to 1)
    sharpen_kernel /= np.sum(sharpen_kernel)
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, sharpen_kernel)

    # Clip the values to [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255).astype('uint8')

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Sharpen Filter')

# * Function to display the emboss filter
def display_emboss_filter(img):
    # Define the Emboss Kernel
    emboss_kernel  = np.array([[0,  0,  0,  0,  0],
                                [0, -2, -1,  0,  0],
                                [0, -1,  1,  1,  0],
                                [0,  0,  1,  2,  0],
                                [0,  0,  0,  0,  0]], np.float32)
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, emboss_kernel)

    # Clip the values to [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255).astype('uint8')

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Emboss Filter')

# * Function to display the invert filter
def display_invert_filter(img):
    # Define the Invert Kernel
    invert_kernel = np.array([[-1]])
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, invert_kernel)

    filtered_image = 255 - img

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Invert Filter')

# * Function to prompt the user for a custom kernel
def get_custom_kernel():
    while True:
        try:
            # Ask the user for the size of the kernel (assumed to be square)
            size = int(input("Enter the size of the kernel: "))
            if size <= 0:
                print("The size must be a positive integer.")
                continue

            # Initialize an empty list to hold the kernel rows
            kernel = []
            print("Enter the kernel values row by row, separated by spaces:")

            for i in range(size):
                while True:
                    # Split the input row into individual numbers
                    row_input = input(f"Row {i+1}: ").strip().split()
                    
                    if len(row_input) != size:
                        print(f"Please enter exactly {size} values separated by spaces.")
                        continue
                    
                    try:
                        # Convert row values to float and append to the kernel
                        row = list(map(float, row_input))
                        kernel.append(row)
                        break
                    except ValueError:
                        print("Please enter valid numbers.")

            # Convert the list of lists into a numpy array and return
            return np.array(kernel, dtype=np.float32)

        except ValueError:
            print("Please enter a valid integer for the size.")

# * Function to display the custom filter
def display_custom_filter(img):
    # Get the custom kernel from the user
    custom_kernel = get_custom_kernel()
    
    # Apply the filter
    filtered_image = apply_kernel_filter(img, custom_kernel)

    # Clip the values to [0, 255]
    filtered_image = np.clip(filtered_image, 0, 255).astype('uint8')

    # Plot the filtered image
    plot_filtered_image(filtered_image, 'Custom Filter')

# * Function to display the menu
def display_menu():
    print("\nMenu:")
    print("1. Original Image")
    print("2. Average Filter")
    print("3. Gaussian Blur")
    print("4. Sharpen Filter")
    print("5. Emboss Filter")
    print("6. Invert Filter")
    print("7. Custom Filter")
    print("8. Exit\n")

# * Function to display the menu and prompt the user for a choice
def get_user_choice():
    # Display the menu
    display_menu()
    # Prompt the user for a choice
    choice = int(input("Enter your choice: "))
    # Return the choice
    return choice

# * Main function
def main():
    # Read the image
    img = cv2.imread('original_image.jpg')

    # Loop until the user exits
    while True:
        # Get the user's choice
        choice = get_user_choice()

        # Display the original filter
        if choice == 1:
            display_original_filter(img)
        # Display the average filter
        elif choice == 2:
            display_average_filter(img)
        # Display the gaussian blur
        elif choice == 3:
            display_gaussian_blur(img)
        # Display the sharpen filter
        elif choice == 4:
            display_sharpen_filter(img)
        # Display the emboss filter
        elif choice == 5:
            display_emboss_filter(img)
        # Display the invert filter
        elif choice == 6:
            display_invert_filter(img)
        # Display the custom filter
        elif choice == 7:
            display_custom_filter(img)
        # Exit the program
        elif choice == 8:
            break
        # Invalid choice
        else:
            print("Invalid choice!")

# * Execute the main function
if __name__ == '__main__':
    main()
