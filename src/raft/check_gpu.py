# check_gpu.py
import torch

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ Sucesso! PyTorch encontrou {gpu_count} GPU(s).")
    print(f"   Dispositivo #0: {gpu_name}")
else:
    print("❌ Falha! PyTorch não conseguiu encontrar uma GPU compatível com CUDA.")