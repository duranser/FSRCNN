# FSRCNN in PyTorch

A compact PyTorch implementation of **Fast Super-Resolution Convolutional Neural Network (FSRCNN)** for single-image super-resolution. The project follows the principal FSRCNN architecture proposed by Dong, Loy, and Tang [[1]](#ref-1), while providing a practical PyTorch training, evaluation, and inference pipeline.

The default experiment uses:

- **Model:** FSRCNN(56,12,4)
- **Scale factor:** ×2
- **Training data:** 91-image dataset [[2]](#ref-2)
- **Evaluation data:** Set5 and Set14 [[3]](#ref-3)[[4]](#ref-4)
- **Reconstructed channel:** MATLAB-compatible luminance \(Y\)
- **Chrominance:** bicubic interpolation
- **Framework:** PyTorch

---

## FSRCNN architecture

<p align="center">
<img width="957" height="381" alt="fsrcnn_architecture" src="https://github.com/user-attachments/assets/ce200f01-f4e5-478a-9932-a19c974605f1" />
</p>

<p align="center">
  <em>Comparison of SRCNN and FSRCNN. Figure is taken from the original FSRCNN paper by Dong, Loy, and Tang [1].</em>
</p>

FSRCNN processes the native low-resolution image directly, rather than first enlarging it with bicubic interpolation. Its hourglass-shaped network contains five stages:

1. **Feature extraction:** a 5×5 convolution maps the LR luminance image to `d` feature maps.
2. **Shrinking:** a 1×1 convolution reduces the feature dimension from `d` to `s`.
3. **Mapping:** `m` consecutive 3×3 convolutional layers perform nonlinear LR-to-HR feature mapping.
4. **Expanding:** a 1×1 convolution restores the feature dimension from `s` to `d`.
5. **Deconvolution:** a learned transposed convolution reconstructs and upsamples the final HR luminance image.

For FSRCNN(56,12,4), the network is:

```text
Conv(5, 56, 1) -> PReLU
Conv(1, 12, 56) -> PReLU
4 x [Conv(3, 12, 12) -> PReLU]
Conv(1, 56, 12) -> PReLU
DeConv(9, 1, 56), stride = scale
```

---

## Configurable parameters

| Argument | Meaning | Default used here |
|---|---|---:|
| `--scale` | Super-resolution scale factor | `2` |
| `--d` | Number of feature maps in feature extraction and expansion | `56` |
| `--s` | Number of feature maps in the shrinking and mapping stages | `12` |
| `--m` | Number of nonlinear mapping layers | `4` |
| `--channels` | Number of reconstructed channels; `1` for luminance or `3` for RGB | `1` |
| `--deconv-kernel` | Spatial kernel size of the transposed convolution | `9` |
| `--deconv-std` | Standard deviation used to initialize the deconvolution weights | `0.001` |
| `--lr-patch-size` | LR training patch width and height | `48` |
| `--repeat` | Dataset repetition factor used for online patch sampling | `100` |
| `--batch-size` | Training batch size | `64` |
| `--epochs` | Number of training epochs | `100` |
| `--lr` | Adam learning rate | `1e-3` |


---

## Differences from the original Caffe implementation

| Aspect | Original Caffe-based FSRCNN | This PyTorch project |
|---|---|---|
| **Framework** | Caffe training pipeline; the original work also used a C++ implementation for runtime evaluation | PyTorch modules, automatic differentiation, DataLoader, CUDA support, and optional AMP |
| **Main architecture** | `Conv(5,d,1) -> Conv(1,s,d) -> m x Conv(3,s,s) -> Conv(1,d,s) -> DeConv(9,1,d)` | The same nominal FSRCNN(56,12,4) topology is preserved |
| **Loss function** | Caffe Euclidean loss, corresponding to a squared-error objective | Mean L1 reconstruction loss |
| **Optimizer** | Stochastic gradient descent with momentum 0.9 | Adam |
| **Training patch size** | Small patches prepared for the original Caffe-valid deconvolution and label alignment | 48×48 LR patches and 96×96 HR targets for ×2 |
| **Patch generation** | Patches generated beforehand and stored for Caffe, commonly in HDF5 format | Random aligned LR/HR patches generated online from complete images |
| **Degradation implementation** | MATLAB-oriented bicubic preprocessing | Pillow bicubic downsampling, applied consistently to the complete image before patch extraction |
| **Y-channel conversion** | MATLAB-compatible limited-range YCbCr luminance | Explicit MATLAB-compatible limited-range Y conversion used in training, validation, testing, and inference |
| **Training dataset** | The best reported paper results used 91-image together with General-100 | The supplied checkpoint uses only 91-image |
| **Evaluation implementation** | MATLAB-oriented PSNR/SSIM, interpolation, color conversion, and border handling | PyTorch PSNR/SSIM with MATLAB-compatible Y, floating-point tensors, and scale-factor border shaving |
| **Chrominance reconstruction** | Super-resolution on Y; Cb and Cr enlarged with bicubic interpolation | The same luminance-only reconstruction and bicubic chrominance procedure |

---

### Comparison with the original paper

| Dataset | Original FSRCNN PSNR | Original FSRCNN SSIM | This project PSNR | This project SSIM | ΔPSNR | ΔSSIM |
|---|---:|---:|---:|---:|---:|---:|
| Set5 ×2 | 37.00 dB | 0.9558 | 36.432 dB | 0.9528 | -0.568 dB | -0.0030 |
| Set14 ×2 | 32.63 dB | 0.9088 | 32.394 dB | 0.9054 | -0.236 dB | -0.0034 |

> The paper [1] values above correspond to the authors' best FSRCNN results trained with both the **91-image** and **General-100** datasets. The supplied project checkpoint was trained only with the **91-image** dataset. The comparison is therefore informative, but not a strictly like-for-like reproduction.

---

## Setup

Create and activate a PyTorch environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


Create the dataset and output directories:

```bash
python -m fsrcnn.utils make-dirs
```

Verify the MATLAB-compatible luminance conversion:

```bash
python -m fsrcnn.utils check-y
```

The check should produce approximately:

```text
black -> 16
white -> 235
red   -> 81
green -> 145
blue  -> 41
```

Place the HR images in:

```text
data/
├── 91-image/
├── Set5/
└── Set14/
```

---

## Train

Train FSRCNN(56,12,4) for ×2 super-resolution:

```bash
bash scripts/train_fsrcnn_x2.sh
```

Equivalent command:

```bash
python train.py \
  --train-dir data/91-image \
  --val-dirs data/Set5 data/Set14 \
  --scale 2 \
  --d 56 \
  --s 12 \
  --m 4 \
  --channels 1 \
  --deconv-kernel 9 \
  --deconv-std 0.001 \
  --lr-patch-size 48 \
  --repeat 100 \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3 \
  --save-dir runs/fsrcnn_deconv_x2
```

The training script saves:

```text
runs/fsrcnn_deconv_x2/
├── best.pth
├── best_psnr.pth
├── best_ssim.pth
├── latest.pth
├── metrics.csv
└── epochs/
```

Inspect all epoch checkpoints:

```bash
python -m fsrcnn.utils inspect-epochs \
  --epoch-dir runs/fsrcnn_deconv_y_x2/epochs
```

> **Validation note:** the default script evaluates Set5 and Set14 during training and selects the best checkpoint from their mean score. For a strictly independent test protocol, provide a separate validation dataset through `--val-dirs` and use Set5 and Set14 only with `test.py`.

---

## Test

Evaluate the best-PSNR checkpoint on Set5 and Set14:

```bash
bash scripts/test_fsrcnn_x2.sh
```

Equivalent command:

```bash
python test.py \
  --checkpoint runs/fsrcnn_deconv_x2/best_psnr.pth \
  --test-dirs data/Set5 data/Set14 \
  --save-images results/fsrcnn_deconv_x2
```

Evaluation is performed on the MATLAB-compatible limited-range luminance channel:

```text
Y = 16 + 65.481 R + 128.553 G + 24.966 B
Y range = [16/255, 235/255]
```

---

## Inference

The inference script expects a **ground-truth image**. It creates an LR input by bicubic downsampling, reconstructs it with FSRCNN, and preserves the original GT dimensions.

```bash
bash scripts/infer_from_gt_x2.sh path/to/ground_truth.png
```

Equivalent command:

```bash
python inference.py \
  --input path/to/ground_truth.png \
  --output results/inference \
  --checkpoint runs/fsrcnn_deconv_x2/best_psnr.pth
```

For a ground-truth image of size WxH, inference saves:

```text
<name>_LR_x2.png                  W/2 x H/2
<name>_LR_Nearest_x2.png          W x H
<name>_GT.png                     W x H
<name>_FSRCNN_Deconv_x2.png       W x H
```

FSRCNN reconstructs only the luminance channel. The Cb and Cr channels are enlarged using bicubic interpolation and combined with the reconstructed Y channel to generate the final RGB output.

---

## Results

The following values were read from the supplied `best_psnr.pth` checkpoint:

- **Checkpoint:** `best_psnr.pth`
- **Epoch:** 97
- **Training dataset:** 91-image
- **Scale:** ×2
- **Model:** FSRCNN(56,12,4)
- **Parameters:** 12,809
- **Mean Set5/Set14 PSNR:** 34.413 dB
- **Mean Set5/Set14 SSIM:** 0.9291

---

## References

1. **FSRCNN paper**  
   C. Dong, C. C. Loy, and X. Tang, “Accelerating the Super-Resolution Convolutional Neural Network,” *European Conference on Computer Vision (ECCV)*, 2016.  
   [Paper](https://arxiv.org/abs/1608.00367) · [Official project page and original Caffe code](https://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)

2. **91-image training dataset**  
   J. Yang, J. Wright, T. S. Huang, and Y. Ma, “Image Super-Resolution via Sparse Representation,” *IEEE Transactions on Image Processing*, vol. 19, no. 11, pp. 2861–2873, 2010.  
   [DOI: 10.1109/TIP.2010.2050625](https://doi.org/10.1109/TIP.2010.2050625)

3. **Set5 benchmark**  
   M. Bevilacqua, A. Roumy, C. Guillemot, and M.-L. Alberi-Morel, “Low-Complexity Single-Image Super-Resolution Based on Nonnegative Neighbor Embedding,” *British Machine Vision Conference (BMVC)*, 2012.  
   [DOI: 10.5244/C.26.135](https://doi.org/10.5244/C.26.135)

4. **Set14 benchmark**  
   R. Zeyde, M. Elad, and M. Protter, “On Single Image Scale-Up Using Sparse-Representations,” in *Curves and Surfaces*, LNCS 6920, pp. 711–730, 2012.  
   [DOI: 10.1007/978-3-642-27413-8_47](https://doi.org/10.1007/978-3-642-27413-8_47)

---

## Acknowledgment

This repository is a PyTorch reimplementation developed for research and educational use. The network figure is taken from the original FSRCNN publication and is included with attribution to the authors.
