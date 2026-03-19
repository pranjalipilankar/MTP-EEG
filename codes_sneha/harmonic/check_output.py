import numpy as np
r = np.load("results_test/subject_003/reconstructed.npy")
g = np.load("results_test/subject_003/ground_truth.npy")
i = np.load("results_test/subject_003/inputs.npy")

print(f"Shapes match: {r.shape == g.shape == i.shape}")  # True
print(f"First segment RMSE: {np.sqrt(np.mean((r[0]-g[0])**2)):.4f}")
