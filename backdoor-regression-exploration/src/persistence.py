"""
src/persistence.py

Saving experiment artifacts under output/<parameter-encoded subfolder>/.

Each run gets its own directory whose name encodes the experiment parameters,
so runs never overwrite each other and the folder listing is self-describing:

    output/
      scheme-hmac_n-6000_seed-0_high-0.95_mode-floor/
        report.txt            # the narrated console report
        params.json           # exact parameters of the run
        results.json          # the test battery metrics, structured
        clean_model.joblib     # the honest regressor
        backdoored_model.joblib# the signature-backdoored wrapper

The backdoored model is saved with whatever key state it holds. For ECDSA you
can call strip_signing_key() before saving to persist a verify-only (deployed)
artifact; see run_demo.py --strip-key.
"""

import json
import os

import joblib


def run_name(params: dict) -> str:
    """Filesystem-safe subfolder name encoding the experiment parameters."""
    return (
        f"scheme-{params['scheme']}"
        f"_n-{params['n']}"
        f"_seed-{params['seed']}"
        f"_high-{params['high_target']:g}"
        f"_mode-{params['shift_mode']}"
    )


def make_output_dir(base_dir: str, params: dict) -> str:
    out = os.path.join(base_dir, run_name(params))
    os.makedirs(out, exist_ok=True)
    return out


def _to_jsonable(obj):
    """Recursively coerce numpy scalars/arrays and bytes into JSON types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_experiment(out_dir, params, results, report_text,
                    clean_model, backdoored_model):
    """Write every artifact for one run. Returns the list of paths written."""
    paths = []

    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    paths.append(report_path)

    params_path = os.path.join(out_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(_to_jsonable(params), f, indent=2)
    paths.append(params_path)

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_to_jsonable(results), f, indent=2)
    paths.append(results_path)

    clean_path = os.path.join(out_dir, "clean_model.joblib")
    joblib.dump(clean_model, clean_path)
    paths.append(clean_path)

    bd_path = os.path.join(out_dir, "backdoored_model.joblib")
    joblib.dump(backdoored_model, bd_path)
    paths.append(bd_path)

    return paths
