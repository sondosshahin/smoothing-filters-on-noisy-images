import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
 

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
    noisy_image[coords[0], coords[1], :] = 255
    
    # Add Pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0
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



def plot_box_filtered(id, image, noisy_images):
  print("Box filter with kernel 3x3 : image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.blur(noisy_img,(3,3))    #box filter
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
  plt.suptitle("Box filter with kernel 3x3")
  plt.show()

  print("Box filter with kernel 5x5 : image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.blur(noisy_img,(5,5))    #box filter
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
  plt.suptitle("Box filter with kernel 5x5")
  plt.show()

  print("Box filter with kernel 7x7 : image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.blur(noisy_img,(7,7))    #box filter
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
  plt.suptitle("Box filter with kernel 7x7")
  plt.show()


def plot_gaussian_filtered(id, image, noisy_images):
  print("Gaussian filter with kernel 3x3: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.GaussianBlur(noisy_img,(3,3),0)
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
  plt.suptitle("Gaussian filter with kernel 3x3")
  plt.show()

  print("Gaussian filter with kernel 5x5: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.GaussianBlur(noisy_img,(5,5),0)
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
  plt.suptitle("Gaussian filter with kernel 5x5")
  plt.show()

  print("Gaussian filter with kernel 7x7: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.GaussianBlur(noisy_img,(7,7),0)
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
  plt.suptitle("Gaussian filter with kernel 7x7")
  plt.show()


def plot_median_filtered(id, image, noisy_images):
  print("Median filter with kernel 3x3: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.medianBlur(noisy_img,3)
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
  plt.suptitle("Median filter with kernel 3x3")
  plt.show()

  print("Median filter with kernel 5x5: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.medianBlur(noisy_img,5)
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
  plt.suptitle("Median filter with kernel 5x5")
  plt.show()

  print("Median filter with kernel 7x7: image ", id)
  for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.medianBlur(noisy_img,7)
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
  plt.suptitle("Median filter with kernel 7x7")
  plt.show()

def plot_bilateral_filtered(id, image, noisy_images):
   print("Bilateral filter: image ", id)
   for i, (key, noisy_img) in enumerate(noisy_images.items(), start=1):    #plot images after filters
    blur = cv.bilateralFilter(noisy_img,9,75,75)
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
   plt.suptitle("Bilateral filter")
   plt.show()



image1 = cv.imread("sample_image1.png")
assert image1 is not None, "file could not be read, check with os.path.exists()"
image2 = cv.imread("sample_image2.png")
assert image1 is not None, "file could not be read, check with os.path.exists()"
image3 = cv.imread("sample_image3.png")
assert image1 is not None, "file could not be read, check with os.path.exists()"
 

noisy_images = add_noise(image1)
display_original_and_noisy(image1, noisy_images)
plot_box_filtered(1, image1, noisy_images)
'''
plot_gaussian_filtered(1, image1, noisy_images)
plot_median_filtered(1, image1, noisy_images)
plot_bilateral_filtered(1, image1, noisy_images)
'''


'''
noisy_images = add_noise(image2)
display_original_and_noisy(image2, noisy_images)
plot_box_filtered(2, image2, noisy_images)
plot_gaussian_filtered(2, image2, noisy_images)
plot_median_filtered(2, image2, noisy_images)
plot_bilateral_filtered(2, image2, noisy_images)


noisy_images = add_noise(image3)
display_original_and_noisy(image3, noisy_images)
plot_box_filtered(3, image3, noisy_images)
plot_gaussian_filtered(3, image3, noisy_images)
plot_median_filtered(3, image3, noisy_images)
plot_bilateral_filtered(3, image3, noisy_images)
'''


