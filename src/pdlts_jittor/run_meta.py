"""运行元信息落盘 + 路径检查 (开源规范要求: 保存实际配置/命令, 缺文件时报清楚缺什么)。

对训练/推理算法零影响, 仅做记录与友好报错。
"""
import json
import os
import sys
import time


def dump_run_meta(out_dir, args, name="config"):
    """把本次运行的参数与命令行写到 out_dir 下。

    生成 `<name>.json`(argparse 参数全量, 含默认值) 与 `command.txt`(原始命令 + 时间 + 相关环境变量)。
    重复运行同一 out_dir 时按时间戳追加, 不覆盖历史记录。
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cfg = {k: v for k, v in sorted(vars(args).items())}
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    env_keys = ("CUDA_VISIBLE_DEVICES", "FP_GRAD_ITERS", "JT_PATCH_STEP", "JITTOR_HOME")
    env = " ".join(f"{k}={os.environ[k]}" for k in env_keys if k in os.environ)
    with open(os.path.join(out_dir, "command.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] cwd={os.getcwd()}\n{env} {sys.executable} {' '.join(sys.argv)}\n\n")


def require_path(path, what, hint):
    """路径不存在时给出明确的中文报错并退出, 而不是在深层 glob/open 处抛难懂异常。"""
    if not path or not os.path.exists(path):
        sys.stderr.write(f"[缺少{what}] 路径不存在: {path!r}\n  修复: {hint}\n")
        sys.exit(2)
