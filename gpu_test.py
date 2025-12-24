import torch
import sys

print(f"Python Version: {sys.version}")
print("--------------------------------------------------")

# Check if CUDA (NVIDIA Driver) is available
if torch.cuda.is_available():
    print("✅ SUCCESS: CUDA is ready!")
    print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM (Memory): {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create a random tensor and move it to GPU to prove it works
    x = torch.rand(5, 3).cuda()
    print("\n   Test Tensor on GPU:\n", x)
else:
    print("❌ ERROR: CUDA is NOT working. You are using CPU.")
    print("   Make sure you activated the environment before running VS Code.")