import onnxruntime as ort
import numpy as np
import os

# Updated Path: pointing to the 'models' folder
model_path = os.path.join("models", "vehicle_classifier.onnx")

if not os.path.exists(model_path):
    print(f"Error: Could not find model at {model_path}. Check your 'models' folder.")
else:
    session = ort.InferenceSession(model_path)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {"input": dummy_input})
    print("ONNX Verification Successful. Output shape:", outputs[0].shape)