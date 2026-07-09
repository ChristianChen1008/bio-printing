# 打印轨迹线宽识别示例

这个示例使用 `segmentation_models.pytorch` 的 `Unet++ + EfficientNet-B1` 做打印轨迹分割，然后用骨架化和距离变换计算线宽。

## 目录结构

```text
line_width_segmentation/
  data/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
  outputs/
  src/
    dataset.py
    train.py
    predict.py
    measure_width.py
  requirements.txt
```

图片和 mask 文件名需要一一对应，例如：

```text
data/train/images/img_001.png
data/train/masks/img_001.png
```

mask 是黑白图：

```text
白色 = 打印轨迹
黑色 = 背景
```

## 安装

```bash
cd line_width_segmentation
pip install -r requirements.txt
```

如果有 NVIDIA GPU，请根据你的 CUDA 版本先安装对应的 PyTorch。

## 训练

```bash
python src/train.py `
  --train-images data/train/images `
  --train-masks data/train/masks `
  --val-images data/val/images `
  --val-masks data/val/masks `
  --epochs 80 `
  --patch-size 768 `
  --batch-size 2 `
  --output outputs/best_model.pth
```

如果显存不够，把 `--patch-size` 改成 `512`，或者把 `--batch-size` 改成 `1`。

## 预测 mask

```bash
python src/predict.py `
  --model outputs/best_model.pth `
  --image data/test/example.png `
  --output-dir outputs `
  --patch-size 768
```

这会生成：

```text
outputs/example/
  pred_mask.png
  probability.png
```

## 计算线宽

如果你知道标定比例，例如 `1 pixel = 2.5 um`：

```bash
python src/measure_width.py `
  --mask outputs/example/pred_mask.png `
  --pixel-size 2.5 `
  --unit um `
  --bin-size 100 `
  --output-dir outputs
```

输出包括平均线宽、最小线宽、最大线宽和标准差。对于长轨迹，还会输出：

```text
outputs/pred_mask/
  summary.csv
  width_profile.csv
  width_bins.csv
  clean_mask.png
  width_overlay.png
  width_heatmap.png
```

其中：

```text
width_profile.csv
  每个中心线点的 x/y 坐标、沿轨迹距离、局部线宽

width_bins.csv
  每隔 100 um 的局部平均线宽、最大线宽、最小线宽和标准差

clean_mask.png
  去除小噪点后的轨迹 mask

width_overlay.png
  轨迹区域和骨架中心线叠加图

width_heatmap.png
  用颜色显示线宽变化的位置图
```

线宽计算方法是：

```text
中心线宽度 = 2 × 中心线像素到轨迹边界的距离
```

## 推荐流程

1. 先精细标注 30-50 张图，跑通训练、预测、测宽。
2. 看预测 mask 的边界是否贴合真实轨迹。
3. 如果边界不稳，增加到 100-200 张图，并覆盖不同线宽、光照、材料和打印参数。
4. 对比 `measure_width.py` 输出和显微镜/ImageJ 人工测量结果，确认误差。
