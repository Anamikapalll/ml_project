import sklearn
import torch

print("\n===============================")
print("  AI ENVIRONMENT IS WORKING!   ")
print("===============================")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"PyTorch Version: {torch.__version__}")
print("===============================\n")


import torch

# This automatically selects CPU since you don't have CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example: Creating a tensor and sending it to your CPU
x = torch.rand(3, 3).to(device)
print(x)
