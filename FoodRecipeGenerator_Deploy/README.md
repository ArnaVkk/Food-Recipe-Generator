# 🍛🍔 Universal Food Recipe Generator

## Deployment Package

This package contains everything needed to run the Food Recipe Generator web application.

**Model Accuracy:** 84.8%  
**Categories:** 181 (80 Indian + 101 Western)  
**Overfitting Gap:** +2.5% (Excellent!)

---

## 📋 Requirements

- Python 3.8 or higher
- NVIDIA GPU with CUDA (optional, CPU works too)
- ~2GB disk space
- ~4GB RAM

---

## 🚀 Quick Start

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python app.py
```

### Step 3: Open in Browser

Go to: **http://127.0.0.1:7860**

---

## 📁 Package Contents

```
FoodRecipeGenerator_Deploy/
├── app.py                 # Main web application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── model/
    └── best_model.pth     # Trained model (84.8% accuracy)
```

---

## 💻 System Requirements

### Minimum (CPU Only):
- 4GB RAM
- Any modern CPU
- Inference: ~2-5 seconds

### Recommended (GPU):
- NVIDIA GPU with 2GB+ VRAM
- CUDA 11.8 or higher
- Inference: <1 second

---

## 🍽️ Supported Foods

### 🇮🇳 Indian Cuisine (80 dishes)
Adhirasam, Aloo Gobi, Aloo Matar, Aloo Methi, Aloo Tikki, Anarsa, Ariselu,
Bandar Laddu, Basundi, Bhatura, Biryani, Boondi, Butter Chicken, Chapati,
Cham Cham, Chana Masala, Chicken Tikka, Chole Bhature, Dahi Bhalla, Dal,
Dal Makhani, Dosa, Gajar Halwa, Gulab Jamun, Idli, Jalebi, Kachori, 
Kadhi Pakoda, Kheer, Kofta, Kulfi, Lassi, Ledikeni, Litti Chokha, 
Malabar Parotta, Malapua, Misi Roti, Modak, Mysore Pak, Naan, Navrattan Korma,
Palak Paneer, Paneer Tikka, Phirni, Poha, Poori, Qubani Ka Meetha, Ras Malai,
Rasgulla, Raita, Rasam, Sandesh, Samosa, Shrikhand, Upma, Vada, and more...

### 🍔 Western/International (101 dishes)
Apple Pie, Baby Back Ribs, Baklava, Beef Carpaccio, Beet Salad, Beignets,
Bibimbap, Bread Pudding, Breakfast Burrito, Bruschetta, Caesar Salad,
Cannoli, Caprese Salad, Carrot Cake, Ceviche, Cheesecake, Chicken Curry,
Chicken Quesadilla, Chicken Wings, Chocolate Cake, Clam Chowder, 
Club Sandwich, Crab Cakes, Creme Brulee, Croque Madame, Cup Cakes,
Donuts, Dumplings, Edamame, Eggs Benedict, Escargots, Falafel, Filet Mignon,
Fish and Chips, Foie Gras, French Fries, French Onion Soup, French Toast,
Fried Calamari, Fried Rice, Frozen Yogurt, Garlic Bread, Gnocchi, Greek Salad,
Grilled Cheese Sandwich, Grilled Salmon, Gyoza, Hamburger, Hot and Sour Soup,
Hot Dog, Huevos Rancheros, Hummus, Ice Cream, Lasagna, Lobster Bisque,
Lobster Roll Sandwich, Macaroni and Cheese, Macarons, Miso Soup, Mussels,
Nachos, Omelette, Onion Rings, Oysters, Pad Thai, Paella, Pancakes,
Panna Cotta, Peking Duck, Pho, Pizza, Pork Chop, Poutine, Prime Rib,
Pulled Pork Sandwich, Ramen, Ravioli, Red Velvet Cake, Risotto, Samosa,
Sashimi, Scallops, Seaweed Salad, Shrimp and Grits, Spaghetti Bolognese,
Spaghetti Carbonara, Spring Rolls, Steak, Strawberry Shortcake, Sushi,
Tacos, Takoyaki, Tiramisu, Tuna Tartare, Waffles

---

## 🔧 Troubleshooting

### "CUDA not available"
- CPU mode will be used automatically
- Inference will be slower but still works

### "Model not found"
- Ensure `model/best_model.pth` exists
- File should be ~100MB

### "Port already in use"
- Close other applications using port 7860
- Or modify the port in `app.py`

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Architecture | EfficientNet-B0 |
| Parameters | ~5.3 million |
| Input Size | 224x224 pixels |
| Training Accuracy | 87.2% |
| Validation Accuracy | 84.8% |
| Overfitting Gap | +2.5% |
| Training Images | 113,900 |

---

## 📝 License

This project uses:
- PyTorch (BSD License)
- Gradio (Apache 2.0)
- EfficientNet pretrained weights (Apache 2.0)

---

**Created:** January 2026  
**Framework:** PyTorch + Gradio
