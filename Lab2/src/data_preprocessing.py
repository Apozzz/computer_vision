from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMAGE_SIZE, BATCH_SIZE, DATA_PATH, VALIDATION_PATH

def create_data_generators():
    """Create train and validation data generators with augmentation."""
    
    # Helps improve the model’s ability to generalize by providing different versions of the input images
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # Normalizes the image pixels (0-255) to (0-1)
        rotation_range=20,           # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,       # Randomly shift the image horizontally
        height_shift_range=0.2,      # Randomly shift the image vertically
        shear_range=0.2,             # Randomly shear the image
        zoom_range=0.2,              # Randomly zoom in/out of the image
        horizontal_flip=True,        # Randomly flip images horizontally
        fill_mode="nearest",         # Fill any missing pixels after transformations
        validation_split=0.2         # Reserve 20% of data for validation
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        VALIDATION_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator