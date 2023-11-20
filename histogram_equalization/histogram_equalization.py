import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the histogram of an image
def compute_histogram(img):
    # Initialize an empty histogram of size 256 (for grayscale images)
    histogram = np.zeros(256)
    # Iterate over all pixel values in the image
    for pixel_value in img.ravel():
        histogram[pixel_value] += 1
    return histogram

# Function to compute the median value from a 3x3 neighborhood
def compute_cmid(img, x, y):
    # Define a 3x3 neighborhood around the pixel (x, y)
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0), (0,  0), (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]
    values = []
    # Collect pixel values from the neighborhood
    for dx, dy in neighbors:
        if 0 <= x+dx < img.shape[0] and 0 <= y+dy < img.shape[1]:
            values.append(img[x+dx, y+dy])
    # Return the median value from the collected values
    return np.median(values)

# Function to perform histogram equalization on an image
def histogram_equalization(img, method=1):
    # Define the number of pixels and max pixel value
    Pmax = img.shape[0] * img.shape[1]
    Cmax = 255
    # Compute the histogram of the image
    H = np.zeros(256)
    for P in range(Pmax):
        H[img.flat[P]] += 1

    # Compute the mid level of the histogram
    Hmid = np.sum(H) / len(H)
    Rver = 0
    Hsum = 0
    L = np.zeros(256)
    R = np.zeros(256)
    Cn = np.zeros(256)
    
    # Compute the L(C) and R(C) values for all intensity levels
    for C in range(Cmax+1):
        L[C] = Rver
        Hsum += H[C]
        while Hsum > Hmid:
            Hsum -= Hmid
            Rver += 1
        R[C] = Rver

        # Compute the new pixel values using the selected method
        if method == 1:
            Cn[C] = (L[C] + R[C]) / 2
        elif method == 2:
            Cn[C] = random.randint(int(L[C]), int(R[C]))

    # Apply the histogram equalization transformation to the image
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if method == 3:
                Cmid = compute_cmid(img, x, y)
                # Assign new pixel values based on the Cmid value and the computed L and R values
                if L[Cmid] <= Cmid <= R[Cmid]:
                    img[x, y] = Cmid
                elif Cmid < L[Cmid]:
                    img[x, y] = L[Cmid]
                else:
                    img[x, y] = R[Cmid]
            else:
                # For methods 1 and 2, directly use the computed Cn values
                img[x, y] = Cn[img[x, y]]

    return img

# Main function to load, process, and visualize the images and histograms
def main():
    # Load the original grayscale image
    img = cv2.imread('original_image.jpg', cv2.IMREAD_GRAYSCALE)
    # Apply histogram equalization to a copy of the original image
    equalized_img = histogram_equalization(np.copy(img), method=1)

    # Compute histograms for the original and equalized images
    original_histogram = compute_histogram(img)
    equalized_histogram = compute_histogram(equalized_img)

    # Visualization using matplotlib
    fig, axarr = plt.subplots(2, 2, figsize=(12, 10))

    # Original image and its histogram
    axarr[0, 0].imshow(img, cmap='gray')
    axarr[0, 0].axis('off')
    axarr[0, 0].set_title('Original Image')

    axarr[0, 1].bar(np.arange(256), original_histogram, color='gray', width=1.0)
    axarr[0, 1].set_xlim([0, 255])
    axarr[0, 1].set_title('Original Histogram')

    # Equalized image and its histogram
    axarr[1, 0].imshow(equalized_img, cmap='gray')
    axarr[1, 0].axis('off')
    axarr[1, 0].set_title('Equalized Image')

    axarr[1, 1].bar(np.arange(256), equalized_histogram, color='gray', width=1.0)
    axarr[1, 1].set_xlim([0, 255])
    axarr[1, 1].set_title('Equalized Histogram')

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

# Execute the main function
if __name__ == '__main__':
    main()