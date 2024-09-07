from matplotlib import pyplot as plt


def plot_image_and_histogram(image, processed_image, titles, axes=None):
    """
    Displays the original image and processed image along with their histograms
    in a side-by-side view using a 2x2 grid.
    
    - image: Original image (grayscale).
    - processed_image: Processed image (grayscale).
    - titles: Tuple containing titles for the original and processed images.
    - axes: External axes, if you want to plot on an existing figure.
    """
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig = None

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title(f'{titles[0]} Image')
    axes[0, 0].axis('off')

    axes[0, 1].hist(image.ravel(), bins=256, range=(0, 256), density=True)
    axes[0, 1].set_title(f'{titles[0]} Histogram')

    axes[1, 0].imshow(processed_image, cmap='gray')
    axes[1, 0].set_title(f'{titles[1]} Image')
    axes[1, 0].axis('off')

    axes[1, 1].hist(processed_image.ravel(), bins=256, range=(0, 256), density=True)
    axes[1, 1].set_title(f'{titles[1]} Histogram')

    if fig is not None:
        plt.tight_layout()
        plt.show()

def plot_template_matching(image, template, matched_image, match_location=None):
    """
    Plots the original image, the template, and the matched image with a rectangle drawn around the best match.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(template, cmap='gray')
    axes[1].set_title("Template Image")
    axes[1].axis('off')

    axes[2].imshow(matched_image, cmap='gray')
    axes[2].set_title(f"Matched Image at {match_location}" if match_location else "Matched Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()