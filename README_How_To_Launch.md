# 🍛 Indian Food Recipe Generator

## Quick Start (For Tomorrow!)

### Option 1: Double-Click Method (Easiest)
1. Go to `C:\Users\91638\Desktop\Capstone Sux`
2. **Double-click** on `Launch_Indian_Food_App.bat`
3. Wait for the server to start
4. Open browser and go to: **http://127.0.0.1:7860**

### Option 2: Manual Method (VS Code Terminal)
1. Open VS Code
2. Open terminal (Ctrl + `)
3. Run these commands:
```powershell
cd "C:\Users\91638\Desktop\Capstone Sux\inversecooking\src"
& "C:/Users/91638/Desktop/Capstone Sux/.venv/Scripts/python.exe" web_app.py
```
4. Open browser: **http://127.0.0.1:7860**

### Option 3: Command Prompt
1. Press `Win + R`, type `cmd`, press Enter
2. Copy and paste:
```
cd "C:\Users\91638\Desktop\Capstone Sux\inversecooking\src"
"C:\Users\91638\Desktop\Capstone Sux\.venv\Scripts\python.exe" web_app.py
```
3. Open browser: **http://127.0.0.1:7860**

---

## How to Stop the Server
- Press `Ctrl + C` in the terminal/command window
- Or just close the terminal window

---

## Troubleshooting

### "Module not found" error
Run this command first:
```
"C:\Users\91638\Desktop\Capstone Sux\.venv\Scripts\pip.exe" install gradio torch torchvision
```

### Port already in use
The app is probably already running. Check your browser at http://127.0.0.1:7860

### Model not found error
Make sure the model file exists at:
`C:\Users\91638\Desktop\Capstone Sux\inversecooking\data\indian_model\best_model.pth`

---

## Project Files

| File | Description |
|------|-------------|
| `Launch_Indian_Food_App.bat` | Double-click to start the app |
| `inversecooking/src/web_app.py` | Main web application |
| `inversecooking/src/finetune_indian_v2.py` | Training script |
| `inversecooking/src/test_indian_model_v2.py` | Testing script |
| `inversecooking/data/indian_model/best_model.pth` | Trained model |

---

## What This App Does
1. Upload a photo of Indian food
2. AI identifies the dish (80 categories supported)
3. Get detailed recipe with ingredients and instructions

**Supported dishes:** Naan, Biryani, Butter Chicken, Dosa, Samosa, Idli, Chole, Dal, Paneer Tikka, Gulab Jamun, and 70+ more!

---

Created: January 17, 2026
