import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
 

def add_gaussian_noise(image, mean=0, var=1):
    # Create Gaussian noise
    gaussian = np.random.normal(mean, var, image.shape).astype(np.uint8)
    noisy_image = cv.add(image, gaussian)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = image.copy()
    # Add Salt noise (white pixels)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # Add Pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

def add_noise(image):
   # Apply different noise levels
    noisy_images = {
    "Gaussian Low": add_gaussian_noise(image, var=0.5),
    "Gaussian Medium": add_gaussian_noise(image, var=1),
    "Gaussian High": add_gaussian_noise(image, var=2),
    "Salt-and-Pepper Low": add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01),
    "Salt-and-Pepper Medium": add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05),
    "Salt-and-Pepper High": add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1),
    }
    return noisy_images

def display_original_and_noisy(image, noisy_images):
   # Display original and noisy images
   plt.figure(figsize=(15, 10))
   plt.subplot(2, 4, 1)
   plt.imshow(image)
   plt.title("Original Image")
   plt.axis("off")
   for i, (key, noisy_img) in enumerate(noisy_images.items(), start=2):    #plot images with noise 
    plt.subplot(2, 4, i)
    plt.imshow(noisy_img)
    plt.title(key)
    plt.axis("off")
   plt.tight_layout()
   plt.show()



def plot_box_filtered(id, image, noisy_images, kernel_size):
  filtered_images = {}
  print(f"Box filter with kernel {kernel_size}x{kernel_size} : image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.blur(noisy_img,(kernel_size,kernel_size))    #box filter
    filtered_images[key] = blur
    plt.subplot(2, 3, i)
    plt.imshow(blur)
    plt.title(key)
    plt.axis("off")
    print(key)
    mse = np.mean((image - noisy_img) ** 2)
    print("mse = ", mse)
    if mse == 0: 
     psnr = float('inf')  # If MSE is 0, PSNR is infinite 
    else: 
     psnr = 10 * np.log10((255 ** 2) / mse) 
    print("PSNR = ", psnr) 
  plt.tight_layout()
  plt.suptitle(f"Box filter with kernel {kernel_size}x{kernel_size}")
  plt.show()
  return filtered_images


def plot_gaussian_filtered(id, image, noisy_images, kernel_size):
  filtered_images = {}
  print(f"Gaussian filter with kernel {kernel_size}x{kernel_size}: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.GaussianBlur(noisy_img,(kernel_size,kernel_size),0)
    filtered_images[key] = blur
    plt.subplot(2, 3, i)
    plt.imshow(blur)
    plt.title(key)
    plt.axis("off")
    print(key)
    mse = np.mean((image - noisy_img) ** 2)
    print("mse = ", mse)
    if mse == 0: 
     psnr = float('inf')  # If MSE is 0, PSNR is infinite 
    else: 
     psnr = 10 * np.log10((255 ** 2) / mse) 
    print("PSNR = ", psnr) 
  plt.tight_layout()
  plt.suptitle(f"Gaussian filter with kernel {kernel_size}x{kernel_size}")
  plt.show()
  return filtered_images


def plot_median_filtered(id, image, noisy_images, kernel_size):
  filtered_images = {}
  print(f"Median filter with kernel {kernel_size}x{kernel_size}: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.medianBlur(noisy_img,kernel_size)
    filtered_images[key] = blur
    plt.subplot(2, 3, i)
    plt.imshow(blur)
    plt.title(key)
    plt.axis("off")
    print(key)
    mse = np.mean((image - noisy_img) ** 2)
    print("mse = ", mse)
    if mse == 0: 
     psnr = float('inf')  # If MSE is 0, PSNR is infinite 
    else: 
     psnr = 10 * np.log10((255 ** 2) / mse) 
    print("PSNR = ", psnr) 
  plt.tight_layout()
  plt.suptitle(f"Median filter with kernel {kernel_size}x{kernel_size}")
  plt.show()
  return filtered_images


def plot_bilateral_filtered(id, image, noisy_images, kernel_size):
   filtered_images = {}
   print(f"Bilateral filter with kernel {kernel_size}x{kernel_size}: image ", id)
   for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.bilateralFilter(noisy_img,kernel_size,75,75)
    filtered_images[key] = blur
    plt.subplot(2, 3, i)
    plt.imshow(blur)
    plt.title(key)
    plt.axis("off")
    print(key)
    mse = np.mean((image - noisy_img) ** 2)
    print("mse = ", mse)
    if mse == 0: 
     psnr = float('inf')  # If MSE is 0, PSNR is infinite 
    else: 
     psnr = 10 * np.log10((255 ** 2) / mse) 
    print("PSNR = ", psnr) 
   plt.tight_layout()
   plt.suptitle(f"Bilateral filter with kernel {kernel_size}x{kernel_size}")
   plt.show()
   return filtered_images



def plot_adaptive_mean_filtered(id, image, noisy_images, kernel_size):
  filtered_images = {}
  print(f"Adaptive Mean filter with kernel {kernel_size}x{kernel_size}: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    # Get image dimensions
    rows, cols = image.shape
    pad_size = kernel_size // 2
    padded_img = cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT, value=0)
    
    # Output image
    result = np.zeros_like(image, dtype=np.float32)
    # Apply adaptive mean filter
    for k in range(rows):
        for j in range(cols):
            # Extract the local region
            local_region = padded_img[k:k + kernel_size, j:j + kernel_size]
            # Compute the mean of the local region
            local_mean = np.mean(local_region)
            # Subtract local mean from the current pixel
            result[k, j] = image[k, j] - local_mean + local_mean
    # Clip values to maintain valid range and convert back to original type
    result = np.clip(result, 0, 255).astype(np.uint8)
    filtered_images[key] = result
    plt.subplot(2, 3, i)
    plt.imshow(result)
    plt.title(key)
    plt.axis("off")
    print(key)
    mse = np.mean((image - noisy_img) ** 2)
    print("mse = ", mse)
    if mse == 0: 
     psnr = float('inf')  # If MSE is 0, PSNR is infinite 
    else: 
     psnr = 10 * np.log10((255 ** 2) / mse) 
    print("PSNR = ", psnr) 
  plt.tight_layout()
  plt.suptitle(f"Adaptive Mean filter with kernel {kernel_size}x{kernel_size}: image")
  plt.show()
  return filtered_images


def plot_adaptive_median_filtered(id, image, noisy_images, kernel_size):
  filtered_images = {}
  print(f"Adaptive Median filter with kernel {kernel_size}x{kernel_size}: image: ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    rows, cols = image.shape
    pad_size = kernel_size // 2
    padded_img = cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_CONSTANT, value=0)
    
    # Output image
    result = np.zeros_like(image, dtype=np.float32)
    # Apply adaptive median filter
    for k in range(rows):
        for j in range(cols):
            local_region = padded_img[k:k + kernel_size, j:j + kernel_size]
            local_mean = np.median(local_region)
            result[k, j] = image[k, j] - local_mean + local_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    filtered_images[key] = result
    plt.subplot(2, 3, i)
    plt.imshow(result)
    plt.title(key)
    plt.axis("off")
    print(key)
    mse = np.mean((image - noisy_img) ** 2)
    print("mse = ", mse)
    if mse == 0: 
     psnr = float('inf')  # If MSE is 0, PSNR is infinite 
    else: 
     psnr = 10 * np.log10((255 ** 2) / mse) 
    print("PSNR = ", psnr) 
  plt.tight_layout()
  plt.suptitle(f"Adaptive Median filter with kernel {kernel_size}x{kernel_size}: image")
  plt.show()
  return filtered_images

def apply_canny( images, size, filter_name):
  for i, (key, image) in enumerate(images.items(), start=1):  
    edge = cv.Canny(image, 100,200)
    plt.subplot(2, 3, i)
    plt.imshow(edge)
    plt.title(key)
    plt.axis("off")
  plt.tight_layout()
  plt.suptitle(f"{filter_name} with kernel {size}x{size}:")
  plt.show()

image1 = cv.imread("sample_image1.png")
assert image1 is not None, "file could not be read, check with os.path.exists()"
# Check if the image is already grayscale
if len(image1.shape) == 3:  # Color image with 3 channels (BGR)
  image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

image2 = cv.imread("sample_image2.png", cv.IMREAD_GRAYSCALE)
assert image2 is not None, "file could not be read, check with os.path.exists()"
if len(image2.shape) == 3:
  image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

image3 = cv.imread("sample_image3.png", cv.IMREAD_GRAYSCALE)
assert image3 is not None, "file could not be read, check with os.path.exists()"
if len(image3.shape) == 3:
  image3 = cv.cvtColor(image3, cv.COLOR_BGR2GRAY)

images = [image1,image2,image3]
kernel_sizes = [3, 5, 7]
i=1
for image in images:
  noisy_images = add_noise(image)
  display_original_and_noisy(image, noisy_images)
  for size in kernel_sizes:
    print("Box filter")
    start_time = time.time()
    filtered = plot_box_filtered(i, image, noisy_images, size)
    end_time = time.time()
    computation = end_time - start_time
    print(f"Kernel size {size}x{size}: {computation:.4f} seconds")
    apply_canny(filtered, size, filter_name="Box filter")

    print("Gaussian filter")
    start_time = time.time()
    filtered = plot_gaussian_filtered(i, image, noisy_images, size)
    end_time = time.time()
    computation = end_time - start_time
    print(f"Kernel size {size}x{size}: {computation:.4f} seconds")
    apply_canny(filtered, size, filter_name="Gaussian filter")

    print("Median filter")
    start_time = time.time()
    filtered = plot_median_filtered(i, image, noisy_images, size)
    end_time = time.time()
    computation = end_time - start_time
    print(f"Kernel size {size}x{size}: {computation:.4f} seconds")
    apply_canny(filtered, size, filter_name="Median filter")

    print("Bilateral filter")
    start_time = time.time()
    filtered = plot_bilateral_filtered(i, image, noisy_images, size)
    end_time = time.time()
    computation = end_time - start_time
    print(f"Kernel size {size}x{size}: {computation:.4f} seconds")
    apply_canny(filtered, size, filter_name="Bilateral filter")

    print("Adaptive mean filter")
    start_time = time.time()
    filtered = plot_adaptive_mean_filtered(i, image, noisy_images, size)
    end_time = time.time()
    computation = end_time - start_time
    print(f"Kernel size {size}x{size}: {computation:.4f} seconds")
    apply_canny(filtered, size, filter_name="Adaptive mean filter")

    print("Adaptive median filter")
    start_time = time.time()
    filtered = plot_adaptive_median_filtered(i, image, noisy_images, size)
    end_time = time.time()
    computation = end_time - start_time
    print(f"Kernel size {size}x{size}: {computation:.4f} seconds")
    apply_canny(filtered, size, filter_name="Adaptive median filter")
  i=i+1

