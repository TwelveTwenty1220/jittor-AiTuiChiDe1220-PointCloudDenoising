import os

import jittor as jt


def save_jittor_state(model, path):
    """把模型 state_dict 保存为 Jittor .pkl 检查点(自动创建父目录)。

    输入 model: jittor nn.Module; path: 输出文件路径。无返回值。
    """
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
    """把 .pkl 检查点按键名加载进模型, 形状不一致时抛 ValueError。

    输入 model: 目标 nn.Module; path: jt.save 保存的 .pkl 路径;
    verbose: 打印 loaded/missing/extra 统计; bake_spectral: 对带 .scale 伴随缓冲的
    谱归一化权重先烘焙 W/max(1, sigma/coeff) 再加载(推理用); coeff: 谱系数。
    返回 (loaded, missing, extra): 三个键名列表, 分别为已加载 / 模型有而 ckpt 无 /
    ckpt 有而模型无。
    """
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
