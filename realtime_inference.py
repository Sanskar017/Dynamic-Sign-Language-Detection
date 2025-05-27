import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model(r'D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Preprocessed data\sign_language_model.h5')

# Load the label classes
class_labels = np.load(r'D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Preprocessed data\classes1.npy', allow_pickle=True)

# Constants
IMG_SIZE = 64
SEQUENCE_LENGTH = 30

# Initialize webcam
cap = cv2.VideoCapture(0)

# Buffer to store frames
frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized_frame = resized_frame / 255.0
    frame_buffer.append(normalized_frame)

    # Keep only the latest 30 frames
    if len(frame_buffer) > SEQUENCE_LENGTH:
        frame_buffer.pop(0)

    # Predict only if we have 30 frames
    if len(frame_buffer) == SEQUENCE_LENGTH:
        input_sequence = np.expand_dims(frame_buffer, axis=0)  # Shape: (1, 30, 64, 64, 3)
        predictions = model.predict(input_sequence)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Display the prediction
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
