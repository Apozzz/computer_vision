import argparse
import os
from image_processing import dilation, erosion, gaussian_blur, load_image, equalize_histogram, otsu_threshold, normalize_image, template_convolution
from plot_utils import plot_image_and_histogram, plot_template_matching

IMAGE_DIR = os.path.join(os.path.dirname(__file__), '../images')

def main(image_name, template_name, method, kernel_size, binary):
    """Main function to apply image processing methods and plot results."""
    image_path = os.path.join(IMAGE_DIR, image_name)
    template_path = os.path.join(IMAGE_DIR, template_name)

    if not os.path.exists(image_path):
        print(f"Error: Image {image_name} not found in {IMAGE_DIR}.")
        return

    if not os.path.exists(template_path):
        print(f"Error: Template {template_name} not found in {IMAGE_DIR}.")
        return
    
    """Main function to apply image processing methods and plot results."""
    mode = 'binary' if binary else 'grayscale'
    image = load_image(image_path, mode=mode)

    if method == 'equalization':
        equalized_image = equalize_histogram(image)
        plot_image_and_histogram(image, equalized_image, titles=('Original', 'Equalized'))

    elif method == 'otsu':
        otsu_image, threshold = otsu_threshold(image)
        plot_image_and_histogram(image, otsu_image, titles=('Original', f'Otsu (Threshold {threshold})'))

    elif method == 'normalization':
        normalized_image = normalize_image(image)
        plot_image_and_histogram(image, normalized_image, titles=('Original', 'Normalized'))

    #ssd or cross_correlation
    elif method == 'template_convolution':
        image, template, matched_image, best_match = template_convolution(image_path, template_path, metric='ssd')
        plot_template_matching(image, template, matched_image, best_match)

    elif method == 'gaussian_blur':
        blurred_image = gaussian_blur(image, kernel_size)
        plot_image_and_histogram(image, blurred_image, titles=('Original', f'Gaussian Blurred (Kernel Size {kernel_size})'))

    elif method == 'erosion':
        eroded_image = erosion(image, kernel_size=kernel_size, binary=binary)
        plot_image_and_histogram(image, eroded_image, titles=('Original', f'Eroded (Kernel Size {kernel_size})'))

    elif method == 'dilation':
        dilated_image = dilation(image, kernel_size=kernel_size, binary=binary)
        plot_image_and_histogram(image, dilated_image, titles=('Original', f'Dilated (Kernel Size {kernel_size})'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply image processing techniques (Equalization, Otsu, Normalization, Template Matching, Gaussian Blurring, Grayscale Erosion, or Grayscale Dilation)")
    
    parser.add_argument(
        '--image',
        type=str,
        default='image6.jpg',
        help='Name of the input image file (should be in the images/ directory)'
    )

    parser.add_argument(
        '--template', 
        type=str, 
        default='template.jpg',
        help='Name of the template image file (should be in the images/ directory)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='equalization',
        choices=['equalization', 'otsu', 'normalization', 'template_convolution', 'gaussian_blur', 'erosion', 'dilation'],
        help='Image processing method to apply (equalization, otsu, normalization, template_convolution, gaussian_blur, erosion, dilation)'
    )

    parser.add_argument(
        '--kernel_size',
        type=int,
        default=3,
        help='Kernel size for erosion or dilation (must be odd)'
    )

    parser.add_argument(
        '--binary',
        action='store_true',
        help='If set, performs binary erosion or dilation. Otherwise, performs grayscale operations.'
    )
    
    args = parser.parse_args()
    
    main(args.image, args.template, args.method, args.kernel_size, args.binary)
