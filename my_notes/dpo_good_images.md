① 先阅读 Diffusion DPO（以及可能的 NFD）论文
            │
            ▼
② 验证扩散模型是否具有足够随机性
（同一 LQ，sample 多个噪声）
            │
            ▼
③ 对生成结果计算 PSNR、SSIM、LPIPS、CLIP-IQA 等指标
            │
            ▼
④ 判断最好和最差之间差距是否足够大
            │
            ▼
⑤ 用 Sobel 等方法筛选纹理丰富的 GT，构建高质量 RL 数据集
            │
            ▼
⑥ 每个天气任务保留约 7000～8000 对样本
            │
            ▼
⑦ 先在约 100 张图上做验证，每张生成 N 个恢复结果
            │
            ▼
⑧ 自动选出 Winner 和 Loser，构建 Preference Pair
            │
            ▼
⑨ 再把这一流程扩展到整个数据集，开始 DPO/GDPO 训练

dpo流程
Preference

↓

Pair

↓

Winner

Loser

↓

更新Diffusion
