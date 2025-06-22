import torch

torch.set_float32_matmul_precision("highest")  # ensure it's not "medium" or "highest" (they allow mixed precision)

def run_attention():
    B, H, S, D = 32, 68, 128, 64  # Batch, Heads, Seq Len, Head Dim
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # Profiled call
    torch.cuda.synchronize()
    torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

run_attention()
