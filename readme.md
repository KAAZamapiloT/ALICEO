# 🌙 Low-Light Aerial Image Enhancement Using Color–Monochrome Fusion

## 📌 Project Overview

This project implements a **classical image fusion pipeline** for enhancing **low-light aerial images** using a **color camera and a monochrome camera**, inspired by the IEEE research paper:

> **“Low-Light Aerial Imaging With Color and Monochrome Cameras”**  
> Pengwu Yuan, Liqun Lin, Junhong Lin, Yipeng Liao, Tiesong Zhao

Low-light aerial imaging using RGB cameras suffers from **high noise, low contrast, and detail loss**.  
Monochrome cameras, however, provide **higher signal-to-noise ratio (SNR)** but lack color information.

🎯 **Goal:** Combine the strengths of both sensors using **luminance–chrominance fusion** to produce clearer, brighter color images.

---

## 🧠 Core Idea

### 📷 Color Camera (RGB)
- ✅ Preserves color information  
- ❌ Very noisy in low-light conditions  

### 🖤 Monochrome Camera
- ✅ High SNR, strong structural details  
- ❌ No color information  

### 💡 Insight
Use the **monochrome image to guide luminance reconstruction**, while keeping color from the RGB image intact.

---

## 🔧 Methodology Overview

The pipeline follows a **fully explainable, classical image processing approach**:

1. **📥 Input Images**
   - Low-light RGB image  
   - Corresponding low-light monochrome image  

2. **📐 Image Alignment**
   - Align mono image with RGB  
   - Simplified assumption of pre-aligned images (explicitly stated)

3. **🎨 Color Space Conversion**
   - Convert RGB → **YCbCr**
   - Separate luminance (Y) from chrominance (Cb, Cr)

4. **🔀 Luminance Fusion**
   - Fuse RGB luminance with monochrome image:
     ```
     Y_fused = α · Y_rgb + (1 − α) · Y_mono
     ```
   - Monochrome image enhances structure and brightness

5. **✨ Contrast Enhancement**
   - Applied only on fused luminance
   - CLAHE or gamma correction

6. **🔁 Reconstruction**
   - Merge fused luminance with original chrominance
   - Convert back to RGB

7. **📊 Evaluation**
   - Visual comparison
   - Quantitative metrics: **PSNR**, **SSIM**

---




---

## 🧪 Implementation Notes

- 🚫 This project **does NOT reproduce the full deep-learning system** from the original paper  
- ✅ Focuses on a **classical, interpretable fusion strategy**
- 📘 All assumptions and simplifications are **explicitly documented**
- 🎓 Designed to be **defendable in academic evaluation / viva**

---

## 🛠️ Tools & Libraries

- 🐍 Python 3.x  
- 📐 NumPy  
- 🖼️ OpenCV  
- 📈 Matplotlib / scikit-image  

---

## 📈 Results

The fusion-based approach shows:

- 🌟 Improved brightness and visibility  
- 🔇 Reduced noise compared to RGB-only enhancement  
- 🧱 Better edge and structure preservation using mono guidance  

📁 Sample outputs and comparisons are stored in the `results/` directory.

---

## ⚠️ Limitations

- ⏱️ No real-time constraints  
- 🚁 No hardware-level camera synchronization  
- 📏 Simplified alignment assumptions  
- 🤖 No learned noise models  

These limitations are **intentional** and appropriate for an academic image processing project.

---

## 🏁 Conclusion

This project demonstrates that **monochrome-guided luminance fusion** is an effective and explainable method for low-light image enhancement.

It highlights how **classical image processing techniques** can capture the core ideas of modern research while remaining **transparent, reproducible, and academically sound**.

---

## 📚 Reference

P. Yuan, L. Lin, J. Lin, Y. Liao, T. Zhao  
*Low-Light Aerial Imaging With Color and Monochrome Cameras*, IEEE


## 🗂️ Project Structure

```text
low_light_color_mono_fusion/
├── data/                 # Input RGB & mono images
├── src/
│   ├── align.py          # Image alignment logic
│   ├── colorspace.py     # RGB ↔ YCbCr conversion
│   ├── fusion.py         # Luminance fusion algorithm
│   ├── enhance.py        # Contrast enhancement
│   └── main.py           # End-to-end pipeline
├── evaluation/
│   └── metrics.py        # PSNR & SSIM computation
├── results/              # Output images & comparisons
├── report/               # Project report (PDF / LaTeX)
└── README.md

