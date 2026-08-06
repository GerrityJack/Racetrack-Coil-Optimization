"""Compare two first_step_diagnostic.py runs' iteration-1 T-equation
matrices/RHS/solution, layer by layer, bit-level and statistically."""
import sys
import numpy as np

dir_a, dir_b = sys.argv[1], sys.argv[2]

for layer in range(6):
    fa = f"{dir_a}/dump_layer{layer}.npz"
    fb = f"{dir_b}/dump_layer{layer}.npz"
    a = np.load(fa)
    b = np.load(fb)

    same_shape = (a["indptr"].shape == b["indptr"].shape
                 and a["indices"].shape == b["indices"].shape
                 and a["data"].shape == b["data"].shape)
    if not same_shape:
        print(f"layer {layer}: DIFFERENT SPARSITY SHAPE "
              f"(A indptr={a['indptr'].shape} indices={a['indices'].shape} "
              f"data={a['data'].shape}; B indptr={b['indptr'].shape} "
              f"indices={b['indices'].shape} data={b['data'].shape})")
        continue

    same_pattern = (np.array_equal(a["indptr"], b["indptr"])
                    and np.array_equal(a["indices"], b["indices"]))
    data_identical = np.array_equal(a["data"], b["data"])
    rhs_identical = np.array_equal(a["rhs"], b["rhs"])
    sol_identical = np.array_equal(a["sol"], b["sol"])

    data_maxdiff = float(np.max(np.abs(a["data"] - b["data"]))) if not data_identical else 0.0
    data_reldiff = data_maxdiff / (float(np.max(np.abs(a["data"]))) + 1e-300)
    rhs_maxdiff = float(np.max(np.abs(a["rhs"] - b["rhs"]))) if not rhs_identical else 0.0
    rhs_norm = float(np.max(np.abs(a["rhs"]))) + 1e-300
    sol_maxdiff = float(np.max(np.abs(a["sol"] - b["sol"]))) if not sol_identical else 0.0
    sol_norm = float(np.max(np.abs(a["sol"]))) + 1e-300

    print(f"layer {layer}: pattern_identical={same_pattern}  "
          f"data_bit_identical={data_identical} (max|diff|={data_maxdiff:.3e}, "
          f"rel={data_reldiff:.3e})  "
          f"rhs_bit_identical={rhs_identical} (max|diff|={rhs_maxdiff:.3e}, "
          f"rel={rhs_maxdiff/rhs_norm:.3e})  "
          f"sol_bit_identical={sol_identical} (max|diff|={sol_maxdiff:.3e}, "
          f"rel={sol_maxdiff/sol_norm:.3e})  "
          f"nnz={a['data'].shape[0]}")
