checkpoints/ 权重说明
=====================
- bboard_best_bigswa26.pkl        B 榜最终提交所用权重。B 榜复现用这个。
- aboard_best_swa19_bnfix.pkl     A 榜最终提交所用权重。也是 B 榜继续训练的起点。
- SHA256SUMS.txt                  两份权重的 SHA256 校验值。

关于文件格式: 两份检查点(ckpt)均为 Jittor 标准保存格式, 由 jt.save() 生成、jt.load() 读取,
文件扩展名为 .pkl(Jittor 惯例, 等同于 PyTorch 生态的 .pt/.ckpt)。
