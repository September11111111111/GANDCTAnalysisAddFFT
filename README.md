# GANDCTAnalysisAddFFT

本项目基于开源仓库 `RUB-SysSec/GANDCTAnalysis` 进行二次开发，用于对 GAN 生成图像与真实图像进行检测分析。原项目以 DCT 频域特征为主，本版本补充了 FFT/PSD/HFE 特征、Logistic Regression 基线、CNN/MIL-CNN 级联推理流程，以及一个用于演示和批量检测的 Streamlit Web 应用。

## 主要工作

- 增加 FFT、PSD、HFE 频域特征提取流程，用于和原有 DCT 方法对比。
- 增加 FFT-LogReg 与 CNN/MIL-CNN 级联检测流程。
- 增加 `pipeline_patch.py`，支持训练 FFT 模型、训练 CNN、随机路由基线、级联推理、阈值调优和热力图导出。
- 增加 `inference.py`，封装单张图像和批量图像推理逻辑。
- 增加 `app.py`，提供 Streamlit Web 界面，支持单张图片检测、ZIP 批量检测和历史记录。
- 保存部分实验输出、模型文件、日志与结果文件，便于后续复现和展示。

## 目录说明

```text
.
├── app.py                         # Streamlit Web 应用入口
├── inference.py                   # 推理封装，供 Web 应用调用
├── pipeline.py                    # 原始/主流程脚本
├── pipeline_patch.py              # FFT + CNN 级联实验流程
├── fft_psd_hfe_baseline.py        # FFT/PSD/HFE 基线实验
├── fft_psd_hfe_torch.py           # Torch 版本频域实验
├── prepare_dataset.py             # 数据集准备脚本
├── prepare_gray_npy_cache.py      # 灰度 128x128 npy 缓存生成
├── runs/                          # 已上传的非图片实验输出、模型与日志
├── result.zip                     # 实验结果压缩包
├── inference_history.db           # Web 推理历史数据库
└── requirements.txt               # Python 依赖
```

注意：热力图等图片内容没有上传到 GitHub。本地 `runs/stylegan_mil_full/heatmaps/most_fake`、`most_real`、`most_uncertain` 下的 `.png` 文件被排除在本次提交之外。

## 环境安装

建议使用 Python 3.7 到 3.8，并创建独立虚拟环境。

```bash
pip install -r requirements.txt
```

如果只运行 Web 推理页面，还需要安装 Streamlit：

```bash
pip install streamlit pandas joblib
```

## 数据准备

项目的训练和评估脚本默认使用 128x128 灰度图像缓存。可以先准备真实图像和 GAN 生成图像目录，再生成 `.npy` 缓存：

```bash
python prepare_gray_npy_cache.py
```

不同实验脚本的参数可能需要根据本地数据路径调整。删除本地文件前，建议确认数据集原始图片和大体积中间文件是否已另行备份，因为本仓库没有上传图片数据。

## 运行实验流程

完整级联流程示例：

```bash
python pipeline_patch.py --fake /path/to/fake.npy --real /path/to/real.npy --out ./runs/exp1 --cnn_arch mil
```

只使用 SimpleCNN，不导出热力图：

```bash
python pipeline_patch.py --fake /path/to/fake.npy --real /path/to/real.npy --out ./runs/exp1 --cnn_arch simple --skip_heatmap
```

跳过已完成的训练阶段，只做后续评估：

```bash
python pipeline_patch.py --fake /path/to/fake.npy --real /path/to/real.npy --out ./runs/exp1 --skip_train_fft --skip_train_cnn
```

标签约定：

- `0` 表示 GAN 生成图像。
- `1` 表示真实图像。

## 运行 Web 应用

```bash
streamlit run app.py
```

Web 应用支持：

- 上传单张 `jpg/png/bmp/webp` 图片进行检测。
- 上传 ZIP 进行批量检测。
- 使用 FFT-LogReg 和 CNN/MIL-CNN 级联推理。
- 保存推理历史到 `inference_history.db`。

## 已记录结果

### 5000-dataset，epoch=10，batch=32

| Method | Model | Accuracy | Throughput |
| --- | --- | ---: | ---: |
| DCT | log2 | 0.9066 | 837.9 img/s |
| DCT | log | 0.9455 | 866.4 img/s |
| FFT | logistic regression | 0.8544 | 164.83 img/s |

### 2k-subset

| Method | Model | Accuracy | Throughput |
| --- | --- | ---: | ---: |
| DCT | log | 0.9400 | 20000 img/s |
| DCT | log2 | 0.9375 | 26666.7 img/s |
| DCT | cnn | 0.9488 | 796.8 img/s |
| DCT | resnet | 0.9525 | 790.5 img/s |

## 备份说明

本次 GitHub 提交保留了代码、配置、数据库、压缩结果、模型文件和非图片实验输出；图片文件没有上传。如果后续需要完整复现实验展示，请重新生成或单独备份热力图图片和原始数据集。
