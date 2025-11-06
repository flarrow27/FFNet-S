import torch
import os
import argparse
import subprocess

# Import FFNet-S (make sure Networks/FFNet_S.py is in the same directory)
#try:
from Networks.FFNet_S import FFNet_S
#except ImportError:
#    print("Error: Could not import FFNet_S.")
#    print("Please make sure 'Networks/FFNet_S.py' and the 'Networks' folder are accessible.")
#    exit()

# ---
# This wrapper simplifies the model to have only one output (the density map)
# This is CRITICAL for a clean TFLite conversion.
# ---
class FFNet_S_Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        density_map, _ = self.model(x) # We only care about the first output
        return density_map

def convert_model(model_path, crop_size):
    # --- 1. Define File Paths ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please check the path to your 'best_model_mae.pth' file.")
        exit()
    
    onnx_path = "ffnet-s.onnx"
    # We use Float16 quantization for a great balance of speed and accuracy on the Pi
    tflite_path = "ffnet-s_fp16.tflite"

    # --- 2. Load PyTorch Model ---
    print(f"Loading PyTorch model from {model_path}...")
    base_model = FFNet_S()
    base_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    base_model.eval()

    # Wrap the model to simplify its output
    model = FFNet_S_Wrapper(base_model)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Export to ONNX ---
    # Create a dummy input tensor that matches the model's training
    # Batch size 1, 3 color channels, CROP_SIZE x CROP_SIZE
    dummy_input = torch.randn(1, 3, crop_size, crop_size, requires_grad=False)

    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(model,
                      dummy_input,
                      onnx_path,
                      export_params=True,
                      opset_version=18, # A common, stable version
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'])
    print("ONNX export complete.")

    # --- 4. Convert ONNX to TensorFlow SavedModel ---
    print("Importing TensorFlow and ONNX-TF...")
    try:
        import onnx
        import onnx_tf.backend
        import tensorflow as tf
    except ImportError as e:
        print(f"\nError: Missing packages. {e}")
        print("Please run: pip install tensorflow onnx-tf")
        return

    print("Loading ONNX model for conversion...")
    onnx_model = onnx.load(onnx_path)

    print("Converting ONNX to TensorFlow representation...")
    tf_rep = onnx_tf.backend.prepare(onnx_model)

    saved_model_dir = "ffnet_savedmodel"
    print(f"Exporting TensorFlow SavedModel to {saved_model_dir}...")
    tf_rep.export_graph(saved_model_dir)
    print("SavedModel export complete.")

    # --- 5. Convert SavedModel to TFLite (FP16) ---
    print(f"Converting SavedModel to TFLite (FP16) at {tflite_path}...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        
        # Apply FP16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print("TFLite conversion complete.")
    
    except Exception as e:
        print(f"\nError during TFLite conversion: {e}")
        print("TensorFlow conversion failed. Check TensorFlow/TFLite compatibility.")
        return

    print(f"\nSuccess! Your model is ready: {tflite_path}")
    print(f"Now, copy this '{tflite_path}' file to your Raspberry Pi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FFNet-S to TFLite Converter')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained FFNet-S .pth model file (e.g., ckpts/.../best_model_mae.pth)')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='Crop size used during training (512 for SHB, 256 for SHA)')
    args = parser.parse_args()
    
    # Auto-detect crop size if not specified
    model_fname = os.path.basename(args.model_path).lower()
    if args.crop_size == 512 and 'sha' in model_fname:
        args.crop_size = 256
        print(f"Auto-detected SHA model, setting crop_size={args.crop_size}")
    else:
        print(f"Using crop_size={args.crop_size} (default for SHB)")

    convert_model(args.model_path, args.crop_size)