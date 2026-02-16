# 🍛🍔 Smart Ingredient Identifier — Universal Food Recipe Generator

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-84.8%25-brightgreen)
![Categories](https://img.shields.io/badge/Food%20Categories-181-purple)
![License](https://img.shields.io/badge/License-MIT-green)

> 📸 **Upload a food photo → AI identifies the dish → Get a full recipe instantly!**

An AI-powered web application that recognizes **181 food dishes** (80 Indian 🇮🇳 + 101 International 🌍) from images using deep learning, and generates detailed recipes with ingredients and step-by-step cooking instructions.

### ✨ Key Highlights

- 🎯 **84.8% accuracy** on 181 food categories
- 🧠 **EfficientNet-B0** architecture with transfer learning
- 🍛 **80 Indian dishes** — Biryani, Dosa, Butter Chicken, Samosa, Naan, and more
- 🍔 **101 International dishes** — Pizza, Sushi, Tacos, Ramen, Steak, and more
- 🌐 **Gradio web interface** — clean, interactive, and mobile-friendly
- ⚡ Works on both **CPU and GPU**
- 📦 Lightweight deployment — just **~27 MB** total

---

## 🎬 Demo

Upload any food image and get:
- ✅ **Dish name** with confidence score
- ✅ **Cuisine type** (Indian / International)
- ✅ **Full ingredient list**
- ✅ **Step-by-step cooking instructions**

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ArnaVkk/Food-Recipe-Generator.git
cd Food-Recipe-Generator
```

### 2. Install Dependencies
```bash
pip install -r FoodRecipeGenerator_Deploy/requirements.txt
```

### 3. Run the App
```bash
cd FoodRecipeGenerator_Deploy
python app.py
```

### 4. Open in Browser
Go to: **http://127.0.0.1:7860**

Upload a photo of any food dish → Get the recipe instantly!

---

## 📁 Project Structure

```
├── FoodRecipeGenerator_Deploy/     # 🚀 Deployment-ready app
│   ├── app.py                      # Main web application (Gradio)
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Deployment docs
│   └── model/
│       └── best_model.pth          # Trained model (84.8% accuracy)
│
├── inversecooking/                 # 📚 Source code & training pipeline
│   ├── src/                        # Core source code
│   │   ├── model.py                # Model architecture
│   │   ├── train_large_model.py    # Training script
│   │   ├── web_app_large.py        # Alternative web app
│   │   ├── data_loader.py          # Data loading utilities
│   │   ├── modules/                # Neural network modules
│   │   └── utils/                  # Utility functions
│   ├── data/
│   │   ├── demo_imgs/              # Sample test images
│   │   ├── indian_recipes.json     # Recipe database
│   │   └── README.md               # Data documentation
│   └── docs/                       # Project documentation & reports
│
└── README.md                       # This file
```

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python     | 3.8+    | 3.10+       |
| RAM        | 4 GB    | 8 GB        |
| GPU        | Not required (CPU works) | NVIDIA with CUDA |
| Disk Space | ~500 MB | ~500 MB     |

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, EfficientNet-B0
- **Web Framework:** Gradio
- **Image Processing:** Pillow, torchvision
- **Language:** Python 3

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 (Transfer Learning) |
| Parameters | ~5.3 million |
| Input Size | 224 × 224 pixels |
| Training Accuracy | 87.2% |
| Validation Accuracy | 84.8% |
| Overfitting Gap | +2.5% (Excellent) |
| Training Images | 113,900 |
| Total Categories | 181 |

---

## 🍽️ Supported Cuisines

### 🇮🇳 Indian (80 dishes)
Biryani, Butter Chicken, Dosa, Naan, Samosa, Idli, Chole Bhature, Dal Makhani, Gulab Jamun, Paneer Tikka, Palak Paneer, Jalebi, Kheer, Rasgulla, Vada, and 65+ more!

### 🌍 International (101 dishes)
Pizza, Sushi, Hamburger, Tacos, Pad Thai, Ramen, Steak, Cheesecake, Tiramisu, French Fries, Caesar Salad, Lasagna, Paella, Pho, and 87+ more!

---

## 📝 License

This project uses:
- PyTorch (BSD License)
- Gradio (Apache 2.0)
- EfficientNet pretrained weights (Apache 2.0)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## ⭐ Show Your Support

If you found this project useful, give it a ⭐ on GitHub!

---

<p align="center">
  <b>Built with ❤️ using PyTorch + Gradio | January 2026</b>
</p>
