#!/usr/bin/env python3
# Measured (pop, gens, total_evals_per_query, wall_time for N=500 batched rows)
data = [
    (32, 3, 9.3),
    (64, 6, 30.4),
    (128, 10, 78.4),
    (256, 16, 228.3),
    (384, 20, 453.1),
]
print(f"{'pop x gens':<12s}{'evals':>8s}{'batched_N500_time':>20s}{'evals/sec':>12s}{'single_query_time':>20s}")
for pop, gens, t in data:
    evals = pop * gens
    total_vae_calls = evals * 500  # N=500 rows batched together per generation
    rate = total_vae_calls / t
    single_query = evals / rate    # time for ONE control-step query, no row-batching benefit
    label = f"{pop}x{gens}"
    print(f"{label:<12s}{evals:>8d}{t:>19.1f}s{rate:>10.0f}/s{single_query:>18.3f}s")

print()
print("Affordable eval budget at various real control rates (~8500 evals/sec VAE throughput):")
rate_evals_per_sec = 8500
for hz in [50, 20, 10, 5, 2, 1]:
    budget_ms = 1000 / hz
    affordable_evals = rate_evals_per_sec * (budget_ms / 1000)
    print(f"  {hz:>3d} Hz ({budget_ms:>6.0f}ms/step): affordable evals = {affordable_evals:>6.0f}")
