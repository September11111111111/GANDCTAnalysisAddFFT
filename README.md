## 项目简介
本项目基于开源仓库 RUB-SysSec/GANDCTAnalysis 进行二次开发，主要新增了 FFT 频域对比方法、部分模型训练实验，以及训练流程与数据处理方式的优化。
## 我的工作
- 新增 FFT/PSD/HFE 特征提取流程，用于和原有 DCT 方法进行对比
- 补充训练并测试了 cnn、resnet 等模型
- 修改训练模式和数据缓存方式，提升实验运行效率
- 增加日志、批量运行和结果整理脚本，方便实验复现与展示
## How to Run
1. Prepare dataset
2. Generate cached npy files
3. Run baseline FFT experiment
4. Train cnn / resnet models
5. Compare accuracy and throughput
## Results

### 5000-dataset (epoch=10, batch=32)
| Method | Model                | Accuracy | Throughput |
|--------|----------------------|----------|------------|
| DCT    | log2                 | 0.9066   | 837.9 img/s |
| DCT    | log                  | 0.9455   | 866.4 img/s |
| FFT    | logistic regression  | 0.8544   | 164.83 img/s |

### 2k-subset
| Method | Model  | Accuracy | Throughput |
|--------|--------|----------|------------|
| DCT    | log    | 0.9400   | 20000 img/s |
| DCT    | log2   | 0.9375   | 26666.7 img/s |
| DCT    | cnn    | 0.9488   | 796.8 img/s |
| DCT    | resnet | 0.9525   | 790.5 img/s |