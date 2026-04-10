import torch

# 1. Inizializzazione dei tensori (Input e Target non richiedono gradiente)
x = torch.tensor([2.0])
y_true = torch.tensor([5.0])

# 2. Inizializzazione dei Parametri (RICHIEDONO il tracciamento del gradiente)
W = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.5], requires_grad=True)

# 3. Forward Pass
y_pred = W * x + b

# 4. Calcolo della Loss
loss = (y_pred - y_true)**2

# 5. Backward Pass (La magia della differenziazione automatica)
loss.backward()

# 6. Verifica
print(f"Gradiente calcolato da PyTorch per W: {W.grad.item()}")