import numpy as np

def calculate_histogram(image):
    """Calculates and returns the histogram of the image."""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    return hist

def calculate_cdf(hist):
    """Calculates and returns the cumulative distribution function (CDF) from the histogram."""
    return np.cumsum(hist)

def otsu_calculate_threshold(hist, total_pixels):
    """Calculates Otsu's optimal threshold using the histogram."""
    probabilities = hist / total_pixels
    current_max, threshold = 0, 0
    sum_total, sum_foreground = 0, 0
    weight_background, weight_foreground = 0, 0
    
    for i in range(256):
        sum_total += i * probabilities[i]

    for t in range(256):
        weight_background += probabilities[t]
        weight_foreground = 1 - weight_background
        if weight_background == 0 or weight_foreground == 0:
            continue

        sum_foreground += t * probabilities[t]
        mean_background = sum_foreground / weight_background if weight_background != 0 else 0
        mean_foreground = (sum_total - sum_foreground) / weight_foreground if weight_foreground != 0 else 0
        
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = t

    return threshold

def calc_template_convolution(image, template, metric='ssd'):
    """Perform template matching using sum of squared differences (SSD) or cross-correlation."""
    image_height, image_width = image.shape
    template_height, template_width = template.shape

    if template_height > image_height or template_width > image_width:
        raise ValueError("Template size is larger than the image size. Template matching cannot proceed.")
    
    result = np.zeros((image_height - template_height + 1, image_width - template_width + 1))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            region = image[i:i + template_height, j:j + template_width]
            if metric == 'ssd':
                result[i, j] = np.sum((region - template) ** 2)
            elif metric == 'cross_correlation':
                result[i, j] = np.sum(region * template)
            else:
                raise ValueError(f"Unknown metric: {metric}")

    best_match = np.unravel_index(np.argmin(result) if metric == 'ssd' else np.argmax(result), result.shape)
    return best_match, result

def gaussian_kernel(size, sigma=1):
    """Creates a Gaussian kernel manually."""
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def manual_convolution(image, kernel):
    """Manually performs 2D convolution of an image with a given kernel."""
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    blurred_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            blurred_image[i, j] = np.sum(region * kernel)
    
    return blurred_image

def structuring_element(size, fill_value=1):
    """Creates a square structuring element (kernel) for erosion and dilation, with a customizable fill value."""
    return np.full((size, size), fill_value, dtype=np.uint8)

def erosion_convolution(image, kernel_size=3, binary=False):
    """Performs morphological erosion on an image. Supports both binary and grayscale images."""
    kernel = structuring_element(kernel_size)
    image_height, image_width = image.shape
    
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    eroded_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            if binary:
                if np.all(region == kernel):
                    eroded_image[i, j] = 1
                else:
                    eroded_image[i, j] = 0
            else:
                eroded_image[i, j] = np.min(region)

    return eroded_image

def dilation_convolution(image, kernel_size=3, binary=False):
    """Performs morphological dilation on an image. Supports both binary and grayscale images."""
    image_height, image_width = image.shape
    
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    dilated_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            if binary:
                if np.any(region == 1):
                    dilated_image[i, j] = 1
                else:
                    dilated_image[i, j] = 0
            else:
                dilated_image[i, j] = np.max(region)

    return dilated_image