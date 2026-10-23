import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

print("PyTorch version:", torch.__version__)
# simple tensor operation 
if torch.cuda.is_available():
    device = torch.device("cuda")
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    c = a + b
    print("Result of tensor addition on GPU:", c.cpu().numpy())
else:
    print("No GPU available, skipping tensor operation test.")
