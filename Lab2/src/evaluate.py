from data_preprocessing import create_data_generators
from tensorflow.keras.models import load_model
from config import MODEL_SAVE_PATH, BATCH_SIZE

def evaluate_model():
    """Load and evaluate the model on the validation set."""
    
    model = load_model(MODEL_SAVE_PATH)
    
    _, validation_generator = create_data_generators()
    
    validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
    print(f"Validation Accuracy: {validation_accuracy:.2f}")
    return validation_loss, validation_accuracy