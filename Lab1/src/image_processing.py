import cv2
import numpy as np
from calculations import calc_template_convolution, calculate_histogram, calculate_cdf, dilation_convolution, erosion_convolution, gaussian_kernel, manual_convolution, otsu_calculate_threshold

def load_image(image_path, mode='grayscale', threshold=127):
    """
    Loads an image in the specified mode.
    - mode: 'grayscale' (default), 'binary', 'color'
    - threshold: used only if mode is 'binary'
    """
    if mode == 'grayscale':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    elif mode == 'binary':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
        return binary_image

    elif mode == 'color':
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    else:
        raise ValueError(f"Unsupported image mode: {mode}")
    
    return image

def equalize_histogram(image):
    """Applies histogram equalization."""
    hist = calculate_histogram(image)
    cdf = calculate_cdf(hist)
    
    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = (cdf - cdf_min) * 255 / (cdf[-1] - cdf_min)
    cdf_normalized = np.ma.filled(cdf_normalized, 0).astype('uint8')
    
    return cdf_normalized[image]

def otsu_threshold(image):
    """Applies Otsu's thresholding."""
    hist = calculate_histogram(image)
    threshold = otsu_calculate_threshold(hist, image.size)
    _, otsu_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return otsu_image, threshold

def normalize_image(image):
    """Normalizes an image to the full intensity range 0-255."""
    min_val, max_val = np.min(image), np.max(image)
    return np.uint8((image - min_val) * 255 / (max_val - min_val))

def draw_rectangle(image, top_left, template_height, template_width):
    """Draws a rectangle around the matched region, ensuring bounds are respected."""
    matched_image = image.copy()
    
    bottom_right = (top_left[1] + template_width, top_left[0] + template_height)
    
    image_height, image_width = image.shape
    bottom_right = (min(bottom_right[0], image_width), min(bottom_right[1], image_height))
    
    matched_image[top_left[0]:bottom_right[1], top_left[1]] = 255
    matched_image[top_left[0]:bottom_right[1], bottom_right[0] - 1] = 255
    
    matched_image[top_left[0], top_left[1]:bottom_right[0]] = 255
    matched_image[bottom_right[1] - 1, top_left[1]:bottom_right[0]] = 255
    
    return matched_image

def template_convolution(image_path, template_path, metric='ssd'):
    """
    Performs template matching by loading the image and template, calling 
    the calculation function, and drawing the rectangle around the best match.
    """
    image = load_image(image_path)
    template = load_image(template_path)
    
    best_match, _ = calc_template_convolution(image, template, metric=metric)
    matched_image = draw_rectangle(image, best_match, template.shape[0], template.shape[1])
    
    return image, template, matched_image, best_match

def gaussian_blur(image, kernel_size=3, sigma=1):
    """Applies Gaussian averaging (blurring) manually using custom convolution."""
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = manual_convolution(image, kernel)
    return blurred_image
    
def erosion(image, kernel_size=3, binary=False):
    """Performs morphological erosion on a binary image."""
    return erosion_convolution(image, kernel_size, binary)

def dilation(image, kernel_size=3, binary=False):
    """Performs morphological dilation on a binary image."""
    return dilation_convolution(image, kernel_size, binary)
