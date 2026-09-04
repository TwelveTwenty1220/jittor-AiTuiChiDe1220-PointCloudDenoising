import os

import jittor as jt


def save_jittor_state(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    jt.save(model.state_dict(), path)
    print(f"[jt-ckpt] saved -> {path}", flush=True)


def _maybe_bake_spectral_weight(key, value, source, coeff):
    if key != "weight" and not key.endswith(".weight"):
        return value
    scale_key = key[: -len(".weight")] + ".scale" if key != "weight" else "scale"
    if scale_key not in source:
        return value
    scale = source[scale_key]
    factor = jt.maximum(jt.array(1.0), scale.reshape(-1)[0] / coeff)
    return value / factor


def load_jittor_state(model, path, verbose=False, bake_spectral=False, coeff=0.98):
    source = jt.load(path)
    target = model.state_dict()
    loaded = []
    extra = []

    for key, value in source.items():
        if key not in target:
            extra.append(key)
            continue
        if bake_spectral:
            value = _maybe_bake_spectral_weight(key, value, source, coeff)
        if tuple(target[key].shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch {key}: model {tuple(target[key].shape)} vs ckpt {tuple(value.shape)}")
        target[key].assign(value)
        loaded.append(key)

    missing = [key for key in target if key not in source]
    if verbose:
        print(f"[jt-ckpt] loaded={len(loaded)} missing={len(missing)} extra={len(extra)}", flush=True)
        if missing:
            print("  MISSING:")
            for key in missing[:40]:
                print("   ", key, tuple(target[key].shape), flush=True)
        if extra:
            print("  EXTRA:")
            for key in extra[:40]:
                print("   ", key, flush=True)
    return loaded, missing, extra
