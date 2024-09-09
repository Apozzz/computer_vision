from data_preprocessing import create_data_generators
from model import create_cnn_model
from config import BATCH_SIZE, IMAGE_SIZE, EPOCHS, MODEL_SAVE_PATH
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model():
    """Train the CNN model."""
    
    train_generator, validation_generator = create_data_generators()
    
    model = create_cnn_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=train_generator.num_classes)
    
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    return history