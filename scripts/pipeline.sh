#!/usr/bin/env bash
# ControlNetSR 流水线备忘 (按顺序执行)
set -euo pipefail

python scripts/filter_dataset/filter_by_sobel.py --config config/filter_dataset.yaml                           # 1. Sobel 细节筛选 (rain/snow/haze 各 top_k)
python train_controlnet.py                            --config config/train.yaml                                 # 2. SFT 监督训练
python pre_test/test_randomness.py                   --config config/pre_test_randomness.yaml                  # 3. SFT 随机性预检 (可选)
python scripts/build_preference.py                    --config config/build_preference.yaml                     # 4. 离线采样候选 + 4 维 reward
python train_controlnet.py --train_method dpo        --config config/dpo.yaml                                  # 5. DPO 偏好训练
python test.py                                        --config config/test.yaml                                 # 6. 测试集推理
python utils/evaluate.py                              --config config/eval.yaml                                 # 7. PSNR/SSIM/LPIPS/FID 评估
