# =============================================================================
# LARGE FOOD RECIPE GENERATOR - WEB INTERFACE (181 Categories)
# =============================================================================
# Recognizes 80 Indian + 101 Western dishes with 84.8% accuracy!

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

print("🚀 Starting Large Food Recipe Generator...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# =============================================================================
# MODEL - MUST MATCH TRAINING ARCHITECTURE
# =============================================================================
class LargeFoodClassifier(nn.Module):
    def __init__(self, num_classes=181):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.15),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# COMPREHENSIVE RECIPES DATABASE - 181 CATEGORIES
# =============================================================================
RECIPES = {
    # ==================== INDIAN RECIPES (80) ====================
    'indian_adhirasam': {
        'title': '🍪 Adhirasam',
        'cuisine': '🇮🇳 Indian (Tamil Nadu)',
        'ingredients': ['rice flour', 'jaggery', 'cardamom', 'coconut oil', 'sesame seeds'],
        'instructions': ['Dissolve jaggery in water', 'Mix with rice flour to form dough', 'Rest overnight', 'Shape into discs', 'Deep fry until golden']
    },
    'indian_aloo_gobi': {
        'title': '🥔 Aloo Gobi',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['potato', 'cauliflower', 'onion', 'tomato', 'turmeric', 'cumin', 'garam masala'],
        'instructions': ['Cut vegetables into pieces', 'Fry cumin in oil', 'Add onion and tomato', 'Add vegetables and spices', 'Cook covered until tender']
    },
    'indian_aloo_matar': {
        'title': '🥔 Aloo Matar',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['potato', 'green peas', 'onion', 'tomato', 'ginger-garlic', 'garam masala'],
        'instructions': ['Sauté onion until golden', 'Add ginger-garlic paste', 'Add tomato puree', 'Add potatoes and peas', 'Simmer until cooked']
    },
    'indian_aloo_methi': {
        'title': '🥬 Aloo Methi',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['potato', 'fenugreek leaves', 'cumin', 'turmeric', 'red chili', 'amchur'],
        'instructions': ['Dice potatoes', 'Clean methi leaves', 'Fry cumin in oil', 'Add potatoes and spices', 'Add methi and cook']
    },
    'indian_aloo_shimla_mirch': {
        'title': '🫑 Aloo Shimla Mirch',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['potato', 'bell pepper', 'onion', 'cumin', 'coriander', 'turmeric'],
        'instructions': ['Cut vegetables', 'Temper cumin', 'Add onions', 'Add potato and peppers', 'Season and cook']
    },
    'indian_aloo_tikki': {
        'title': '🥔 Aloo Tikki',
        'cuisine': '🇮🇳 Indian (Street Food)',
        'ingredients': ['potato', 'peas', 'bread crumbs', 'chaat masala', 'coriander', 'green chili'],
        'instructions': ['Boil and mash potatoes', 'Mix with peas and spices', 'Shape into patties', 'Shallow fry until crispy', 'Serve with chutney']
    },
    'indian_anarsa': {
        'title': '🍪 Anarsa',
        'cuisine': '🇮🇳 Indian (Maharashtra)',
        'ingredients': ['rice flour', 'jaggery', 'poppy seeds', 'ghee'],
        'instructions': ['Soak rice and grind', 'Mix with jaggery', 'Rest the dough', 'Coat with poppy seeds', 'Deep fry slowly']
    },
    'indian_ariselu': {
        'title': '🍩 Ariselu',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['rice flour', 'jaggery', 'sesame seeds', 'cardamom', 'ghee'],
        'instructions': ['Make jaggery syrup', 'Mix with rice flour', 'Shape into rounds', 'Coat with sesame', 'Deep fry golden']
    },
    'indian_bandar_laddu': {
        'title': '🟡 Bandar Laddu',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['besan', 'sugar', 'ghee', 'cardamom', 'cashews'],
        'instructions': ['Roast besan in ghee', 'Add sugar syrup', 'Mix well', 'Shape into balls', 'Garnish with nuts']
    },
    'indian_basundi': {
        'title': '🥛 Basundi',
        'cuisine': '🇮🇳 Indian (Maharashtra)',
        'ingredients': ['full cream milk', 'sugar', 'cardamom', 'saffron', 'almonds', 'pistachios'],
        'instructions': ['Boil milk and reduce', 'Keep stirring', 'Add sugar and cardamom', 'Add saffron', 'Garnish with nuts']
    },
    'indian_bhatura': {
        'title': '🫓 Bhatura',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['maida', 'yogurt', 'baking powder', 'semolina', 'oil'],
        'instructions': ['Mix flour with yogurt', 'Add baking powder', 'Knead soft dough', 'Rest 2 hours', 'Roll and deep fry']
    },
    'indian_bhindi_masala': {
        'title': '🫛 Bhindi Masala',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['okra', 'onion', 'tomato', 'cumin', 'coriander', 'amchur', 'red chili'],
        'instructions': ['Slice okra', 'Fry until crisp', 'Make onion-tomato masala', 'Add okra', 'Season with amchur']
    },
    'indian_biryani': {
        'title': '🍚 Biryani',
        'cuisine': '🇮🇳 Indian (Hyderabad)',
        'ingredients': ['basmati rice', 'chicken/mutton', 'yogurt', 'biryani masala', 'saffron', 'onion', 'ghee', 'mint'],
        'instructions': ['Marinate meat in yogurt and spices', 'Parboil rice', 'Layer meat and rice', 'Add saffron milk', 'Dum cook 30-40 mins']
    },
    'indian_boondi': {
        'title': '🟠 Boondi',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['besan', 'water', 'sugar', 'cardamom', 'saffron'],
        'instructions': ['Make thin batter', 'Drop through ladle into oil', 'Fry crispy', 'Soak in sugar syrup', 'Drain and serve']
    },
    'indian_butter_chicken': {
        'title': '🍗 Butter Chicken',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['chicken', 'tomatoes', 'butter', 'cream', 'garam masala', 'kasuri methi', 'ginger-garlic'],
        'instructions': ['Marinate and grill chicken', 'Make tomato gravy', 'Add butter and cream', 'Add chicken pieces', 'Finish with kasuri methi']
    },
    'indian_chak_hao_kheer': {
        'title': '🍚 Chak Hao Kheer',
        'cuisine': '🇮🇳 Indian (Manipur)',
        'ingredients': ['black rice', 'milk', 'sugar', 'cardamom', 'cashews'],
        'instructions': ['Soak black rice', 'Cook in milk', 'Add sugar', 'Simmer until thick', 'Garnish with nuts']
    },
    'indian_cham_cham': {
        'title': '🍬 Cham Cham',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['chhena', 'sugar', 'cardamom', 'coconut', 'milk'],
        'instructions': ['Make chhena from milk', 'Shape into ovals', 'Cook in sugar syrup', 'Cool and stuff with cream', 'Roll in coconut']
    },
    'indian_chana_masala': {
        'title': '🫘 Chana Masala',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['chickpeas', 'onion', 'tomato', 'chana masala', 'amchur', 'ginger'],
        'instructions': ['Soak and cook chickpeas', 'Make onion-tomato gravy', 'Add chana masala', 'Add chickpeas', 'Simmer with amchur']
    },
    'indian_chapati': {
        'title': '🫓 Chapati',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['whole wheat flour', 'water', 'salt', 'oil'],
        'instructions': ['Knead soft dough', 'Rest 30 mins', 'Divide into balls', 'Roll into circles', 'Cook on tawa']
    },
    'indian_chhena_kheeri': {
        'title': '🥛 Chhena Kheeri',
        'cuisine': '🇮🇳 Indian (Odisha)',
        'ingredients': ['chhena', 'milk', 'sugar', 'cardamom', 'saffron'],
        'instructions': ['Make fresh chhena', 'Reduce milk', 'Add chhena pieces', 'Sweeten with sugar', 'Add flavoring']
    },
    'indian_chicken_razala': {
        'title': '🍗 Chicken Razala',
        'cuisine': '🇮🇳 Indian (Kolkata)',
        'ingredients': ['chicken', 'yogurt', 'onion', 'ginger-garlic', 'white pepper', 'cream'],
        'instructions': ['Marinate chicken', 'Fry onions until brown', 'Add chicken and cook', 'Add yogurt', 'Finish with cream']
    },
    'indian_chicken_tikka': {
        'title': '🍢 Chicken Tikka',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['chicken', 'yogurt', 'tikka masala', 'lemon', 'ginger-garlic', 'kashmiri chili'],
        'instructions': ['Cut chicken into pieces', 'Marinate in yogurt and spices', 'Thread on skewers', 'Grill until charred', 'Serve with chutney']
    },
    'indian_chicken_tikka_masala': {
        'title': '🍛 Chicken Tikka Masala',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['chicken tikka', 'tomato', 'cream', 'onion', 'garam masala', 'kasuri methi'],
        'instructions': ['Make chicken tikka', 'Prepare tomato gravy', 'Add cream', 'Add tikka pieces', 'Garnish with cream']
    },
    'indian_chikki': {
        'title': '🥜 Chikki',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['peanuts', 'jaggery', 'ghee', 'cardamom'],
        'instructions': ['Roast peanuts', 'Melt jaggery', 'Mix together', 'Spread on plate', 'Cut into pieces']
    },
    'indian_daal_baati_churma': {
        'title': '🫓 Dal Baati Churma',
        'cuisine': '🇮🇳 Indian (Rajasthan)',
        'ingredients': ['wheat flour', 'ghee', 'mixed dal', 'sugar', 'cardamom'],
        'instructions': ['Make baati dough', 'Bake until golden', 'Cook dal', 'Crush baati for churma', 'Serve together']
    },
    'indian_daal_puri': {
        'title': '🫓 Dal Puri',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['wheat flour', 'chana dal', 'cumin', 'fennel', 'ginger'],
        'instructions': ['Cook and mash dal', 'Season with spices', 'Make dough with dal', 'Roll into puris', 'Deep fry']
    },
    'indian_dal_makhani': {
        'title': '🫘 Dal Makhani',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['black urad dal', 'rajma', 'butter', 'cream', 'tomato', 'garam masala'],
        'instructions': ['Soak dals overnight', 'Pressure cook until soft', 'Make tomato gravy', 'Simmer with dal', 'Add butter and cream']
    },
    'indian_dal_tadka': {
        'title': '🍲 Dal Tadka',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['toor dal', 'tomato', 'garlic', 'cumin', 'ghee', 'red chili', 'curry leaves'],
        'instructions': ['Cook dal until soft', 'Make tadka with ghee and spices', 'Add tomatoes', 'Pour over dal', 'Garnish with cilantro']
    },
    'indian_dharwad_pedha': {
        'title': '🟤 Dharwad Pedha',
        'cuisine': '🇮🇳 Indian (Karnataka)',
        'ingredients': ['khoya', 'sugar', 'cardamom', 'ghee'],
        'instructions': ['Cook khoya until grainy', 'Add sugar', 'Mix well', 'Shape into pedhas', 'Cool and serve']
    },
    'indian_doodhpak': {
        'title': '🥛 Doodhpak',
        'cuisine': '🇮🇳 Indian (Gujarat)',
        'ingredients': ['milk', 'rice', 'sugar', 'cardamom', 'saffron', 'nuts'],
        'instructions': ['Boil milk', 'Add soaked rice', 'Cook slowly', 'Add sugar and saffron', 'Garnish with nuts']
    },
    'indian_double_ka_meetha': {
        'title': '🍞 Double Ka Meetha',
        'cuisine': '🇮🇳 Indian (Hyderabad)',
        'ingredients': ['bread', 'milk', 'sugar', 'ghee', 'cardamom', 'saffron', 'nuts'],
        'instructions': ['Fry bread in ghee', 'Make sweet milk', 'Soak bread in milk', 'Add saffron', 'Garnish with nuts']
    },
    'indian_dum_aloo': {
        'title': '🥔 Dum Aloo',
        'cuisine': '🇮🇳 Indian (Kashmir)',
        'ingredients': ['baby potatoes', 'yogurt', 'tomato', 'ginger', 'fennel', 'garam masala'],
        'instructions': ['Fry baby potatoes', 'Make yogurt gravy', 'Add potatoes', 'Dum cook covered', 'Garnish with cilantro']
    },
    'indian_gajar_ka_halwa': {
        'title': '🥕 Gajar Ka Halwa',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['carrots', 'milk', 'sugar', 'ghee', 'cardamom', 'cashews', 'raisins'],
        'instructions': ['Grate carrots', 'Cook in milk', 'Reduce until thick', 'Add ghee and sugar', 'Garnish with nuts']
    },
    'indian_gavvalu': {
        'title': '🍪 Gavvalu',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['rice flour', 'sugar', 'ghee', 'cardamom'],
        'instructions': ['Make sugar syrup', 'Mix with rice flour', 'Shape into shells', 'Deep fry', 'Cool and store']
    },
    'indian_ghevar': {
        'title': '🍯 Ghevar',
        'cuisine': '🇮🇳 Indian (Rajasthan)',
        'ingredients': ['flour', 'ghee', 'milk', 'sugar', 'saffron', 'rabri'],
        'instructions': ['Make thin batter', 'Pour in special mold', 'Fry in ghee', 'Soak in syrup', 'Top with rabri']
    },
    'indian_gulab_jamun': {
        'title': '🍩 Gulab Jamun',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['khoya', 'flour', 'cardamom', 'sugar', 'rose water', 'ghee'],
        'instructions': ['Make soft dough', 'Shape into balls', 'Fry on low heat', 'Make sugar syrup', 'Soak jamuns in syrup']
    },
    'indian_imarti': {
        'title': '🟠 Imarti',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['urad dal', 'saffron', 'sugar', 'cardamom', 'ghee'],
        'instructions': ['Soak and grind dal', 'Make thick batter', 'Pipe into flower shape', 'Fry crispy', 'Soak in syrup']
    },
    'indian_jalebi': {
        'title': '🟡 Jalebi',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['flour', 'yogurt', 'sugar', 'saffron', 'cardamom'],
        'instructions': ['Ferment batter', 'Make sugar syrup', 'Pipe spirals into oil', 'Fry until crispy', 'Dip in syrup']
    },
    'indian_kachori': {
        'title': '🥟 Kachori',
        'cuisine': '🇮🇳 Indian (Rajasthan)',
        'ingredients': ['flour', 'moong dal', 'fennel', 'red chili', 'asafoetida'],
        'instructions': ['Make filling with dal', 'Prepare dough', 'Stuff dough with filling', 'Roll gently', 'Deep fry']
    },
    'indian_kadai_paneer': {
        'title': '🧀 Kadai Paneer',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['paneer', 'bell peppers', 'onion', 'tomato', 'kadai masala', 'kasuri methi'],
        'instructions': ['Fry paneer cubes', 'Make tomato gravy', 'Add bell peppers', 'Add kadai masala', 'Finish with kasuri methi']
    },
    'indian_kadhi_pakoda': {
        'title': '🍲 Kadhi Pakoda',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['yogurt', 'besan', 'onion', 'turmeric', 'cumin', 'fenugreek'],
        'instructions': ['Make yogurt-besan mixture', 'Make besan pakodas', 'Cook kadhi', 'Add pakodas', 'Temper with spices']
    },
    'indian_kajjikaya': {
        'title': '🥟 Kajjikaya',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['flour', 'coconut', 'sugar', 'cardamom', 'ghee'],
        'instructions': ['Make sweet filling', 'Prepare dough', 'Shape like half-moon', 'Seal edges', 'Deep fry']
    },
    'indian_kakinada_khaja': {
        'title': '🍬 Kakinada Khaja',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['flour', 'sugar', 'ghee', 'cardamom'],
        'instructions': ['Make layered dough', 'Roll and fold', 'Deep fry', 'Soak in syrup', 'Dry and serve']
    },
    'indian_kalakand': {
        'title': '🍬 Kalakand',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['milk', 'sugar', 'cardamom', 'pistachios'],
        'instructions': ['Curdle milk', 'Cook with sugar', 'Stir continuously', 'Set in tray', 'Cut into pieces']
    },
    'indian_karela_bharta': {
        'title': '🥒 Karela Bharta',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['bitter gourd', 'onion', 'tomato', 'spices', 'mustard oil'],
        'instructions': ['Roast karela', 'Mash roughly', 'Make onion base', 'Add karela', 'Season well']
    },
    'indian_kofta': {
        'title': '🍡 Kofta',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['paneer/potato', 'onion', 'tomato', 'cream', 'garam masala'],
        'instructions': ['Make kofta balls', 'Deep fry', 'Prepare gravy', 'Add koftas', 'Serve hot']
    },
    'indian_kuzhi_paniyaram': {
        'title': '⚫ Kuzhi Paniyaram',
        'cuisine': '🇮🇳 Indian (Tamil)',
        'ingredients': ['idli batter', 'onion', 'curry leaves', 'green chili', 'mustard'],
        'instructions': ['Add tempering to batter', 'Pour in paniyaram pan', 'Cook both sides', 'Serve with chutney']
    },
    'indian_lassi': {
        'title': '🥛 Lassi',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['yogurt', 'sugar/salt', 'cardamom', 'rose water', 'ice'],
        'instructions': ['Blend yogurt smooth', 'Add sweetener', 'Add flavoring', 'Blend with ice', 'Top with cream']
    },
    'indian_ledikeni': {
        'title': '🟤 Ledikeni',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['chhena', 'flour', 'sugar', 'cardamom', 'saffron'],
        'instructions': ['Make chhena', 'Shape into ovals', 'Deep fry', 'Soak in syrup', 'Serve chilled']
    },
    'indian_litti_chokha': {
        'title': '🫓 Litti Chokha',
        'cuisine': '🇮🇳 Indian (Bihar)',
        'ingredients': ['wheat flour', 'sattu', 'brinjal', 'tomato', 'ghee', 'ajwain'],
        'instructions': ['Make sattu filling', 'Stuff in dough', 'Bake on coals', 'Make chokha', 'Serve with ghee']
    },
    'indian_lyangcha': {
        'title': '🍬 Lyangcha',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['chhena', 'flour', 'sugar', 'cardamom'],
        'instructions': ['Mix chhena and flour', 'Shape into cylinders', 'Deep fry', 'Soak in syrup', 'Serve cold']
    },
    'indian_maach_jhol': {
        'title': '🐟 Maach Jhol',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['fish', 'potato', 'tomato', 'turmeric', 'cumin', 'green chili'],
        'instructions': ['Fry fish lightly', 'Make gravy', 'Add potatoes', 'Add fish', 'Simmer gently']
    },
    'indian_makki_di_roti_sarson_da_saag': {
        'title': '🌿 Makki Roti Sarson Saag',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['corn flour', 'mustard greens', 'spinach', 'ginger', 'ghee', 'jaggery'],
        'instructions': ['Cook greens', 'Blend coarsely', 'Season with spices', 'Make corn rotis', 'Serve with ghee']
    },
    'indian_malapua': {
        'title': '🥞 Malapua',
        'cuisine': '🇮🇳 Indian (Bihar)',
        'ingredients': ['flour', 'milk', 'sugar', 'fennel', 'cardamom'],
        'instructions': ['Make thick batter', 'Add fennel', 'Fry like pancakes', 'Soak in syrup', 'Serve warm']
    },
    'indian_misi_roti': {
        'title': '🫓 Misi Roti',
        'cuisine': '🇮🇳 Indian (Punjab)',
        'ingredients': ['wheat flour', 'besan', 'onion', 'coriander', 'ajwain', 'green chili'],
        'instructions': ['Mix flours', 'Add onion and spices', 'Knead dough', 'Roll rotis', 'Cook on tawa']
    },
    'indian_misti_doi': {
        'title': '🥛 Mishti Doi',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['milk', 'sugar/jaggery', 'yogurt culture'],
        'instructions': ['Reduce milk', 'Add caramelized sugar', 'Cool to lukewarm', 'Add culture', 'Set in clay pots']
    },
    'indian_modak': {
        'title': '🥟 Modak',
        'cuisine': '🇮🇳 Indian (Maharashtra)',
        'ingredients': ['rice flour', 'coconut', 'jaggery', 'cardamom', 'ghee'],
        'instructions': ['Make sweet filling', 'Prepare rice dough', 'Shape into modaks', 'Steam until cooked', 'Brush with ghee']
    },
    'indian_mysore_pak': {
        'title': '🟡 Mysore Pak',
        'cuisine': '🇮🇳 Indian (Karnataka)',
        'ingredients': ['besan', 'ghee', 'sugar', 'cardamom'],
        'instructions': ['Make sugar syrup', 'Add roasted besan', 'Pour ghee continuously', 'Mix vigorously', 'Set and cut']
    },
    'indian_naan': {
        'title': '🫓 Naan',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['flour', 'yeast', 'yogurt', 'garlic', 'butter', 'cilantro'],
        'instructions': ['Make soft dough', 'Let rise 2 hours', 'Roll into ovals', 'Cook in tandoor/pan', 'Brush with butter']
    },
    'indian_navrattan_korma': {
        'title': '🥗 Navrattan Korma',
        'cuisine': '🇮🇳 Indian (Mughlai)',
        'ingredients': ['mixed vegetables', 'cream', 'cashews', 'raisins', 'paneer', 'saffron'],
        'instructions': ['Cook vegetables', 'Make cashew cream gravy', 'Add vegetables', 'Add cream and nuts', 'Garnish with saffron']
    },
    'indian_palak_paneer': {
        'title': '🥬 Palak Paneer',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['spinach', 'paneer', 'onion', 'tomato', 'cream', 'garam masala'],
        'instructions': ['Blanch and puree spinach', 'Fry paneer cubes', 'Make onion base', 'Add spinach puree', 'Add paneer and cream']
    },
    'indian_paneer_butter_masala': {
        'title': '🧀 Paneer Butter Masala',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['paneer', 'tomato', 'butter', 'cream', 'kasuri methi', 'garam masala'],
        'instructions': ['Fry paneer cubes', 'Make tomato gravy', 'Add butter and cream', 'Add paneer', 'Garnish with cream']
    },
    'indian_phirni': {
        'title': '🥛 Phirni',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['rice', 'milk', 'sugar', 'cardamom', 'saffron', 'pistachios'],
        'instructions': ['Soak and grind rice', 'Cook in milk', 'Stir until thick', 'Add sugar and saffron', 'Set in clay pots']
    },
    'indian_pithe': {
        'title': '🥟 Pithe',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['rice flour', 'coconut', 'jaggery', 'khoya'],
        'instructions': ['Make rice dough', 'Prepare filling', 'Shape pithe', 'Steam or fry', 'Serve with syrup']
    },
    'indian_poha': {
        'title': '🍚 Poha',
        'cuisine': '🇮🇳 Indian (Maharashtra)',
        'ingredients': ['flattened rice', 'onion', 'potato', 'peanuts', 'turmeric', 'curry leaves'],
        'instructions': ['Rinse and drain poha', 'Fry peanuts', 'Sauté onion and potato', 'Add poha and turmeric', 'Garnish with coriander']
    },
    'indian_poornalu': {
        'title': '🥟 Poornalu',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['rice flour', 'chana dal', 'jaggery', 'cardamom', 'coconut'],
        'instructions': ['Make sweet filling', 'Prepare rice dough', 'Stuff with filling', 'Deep fry', 'Serve warm']
    },
    'indian_pootharekulu': {
        'title': '📜 Pootharekulu',
        'cuisine': '🇮🇳 Indian (Andhra)',
        'ingredients': ['rice starch sheets', 'ghee', 'sugar', 'cardamom'],
        'instructions': ['Make thin rice sheets', 'Layer with ghee', 'Sprinkle sugar', 'Fold carefully', 'Cut and serve']
    },
    'indian_qubani_ka_meetha': {
        'title': '🍑 Qubani Ka Meetha',
        'cuisine': '🇮🇳 Indian (Hyderabad)',
        'ingredients': ['dried apricots', 'sugar', 'cream', 'almonds', 'cardamom'],
        'instructions': ['Soak apricots', 'Cook until soft', 'Puree smooth', 'Add sugar', 'Serve with cream']
    },
    'indian_rabri': {
        'title': '🥛 Rabri',
        'cuisine': '🇮🇳 Indian (North)',
        'ingredients': ['full cream milk', 'sugar', 'cardamom', 'saffron', 'pistachios'],
        'instructions': ['Boil milk', 'Collect cream layers', 'Reduce milk', 'Add sugar', 'Mix with cream layers']
    },
    'indian_ras_malai': {
        'title': '🥛 Ras Malai',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['chhena', 'milk', 'sugar', 'cardamom', 'saffron', 'pistachios'],
        'instructions': ['Make chhena', 'Shape into discs', 'Cook in sugar syrup', 'Make flavored milk', 'Soak discs in milk']
    },
    'indian_rasgulla': {
        'title': '⚪ Rasgulla',
        'cuisine': '🇮🇳 Indian (Bengal/Odisha)',
        'ingredients': ['chhena', 'sugar', 'rose water', 'cardamom'],
        'instructions': ['Make fresh chhena', 'Knead until smooth', 'Shape into balls', 'Cook in sugar syrup', 'Cool in syrup']
    },
    'indian_sandesh': {
        'title': '🍬 Sandesh',
        'cuisine': '🇮🇳 Indian (Bengal)',
        'ingredients': ['chhena', 'sugar', 'cardamom', 'pistachios', 'saffron'],
        'instructions': ['Make fresh chhena', 'Mix with sugar', 'Cook briefly', 'Shape in molds', 'Garnish with nuts']
    },
    'indian_shankarpali': {
        'title': '🍪 Shankarpali',
        'cuisine': '🇮🇳 Indian (Maharashtra)',
        'ingredients': ['flour', 'sugar', 'ghee', 'cardamom', 'sesame'],
        'instructions': ['Make sweet dough', 'Roll and cut diamonds', 'Deep fry', 'Cool completely', 'Store airtight']
    },
    'indian_sheer_korma': {
        'title': '🥛 Sheer Korma',
        'cuisine': '🇮🇳 Indian (Hyderabad)',
        'ingredients': ['vermicelli', 'milk', 'ghee', 'dates', 'nuts', 'saffron'],
        'instructions': ['Fry vermicelli in ghee', 'Boil milk', 'Add vermicelli', 'Add dry fruits', 'Simmer until thick']
    },
    'indian_sheera': {
        'title': '🟡 Sheera',
        'cuisine': '🇮🇳 Indian (Maharashtra)',
        'ingredients': ['semolina', 'ghee', 'sugar', 'cardamom', 'saffron', 'nuts'],
        'instructions': ['Roast semolina in ghee', 'Add water', 'Cook until thick', 'Add sugar', 'Garnish with nuts']
    },
    'indian_shrikhand': {
        'title': '🥛 Shrikhand',
        'cuisine': '🇮🇳 Indian (Maharashtra/Gujarat)',
        'ingredients': ['hung curd', 'sugar', 'cardamom', 'saffron', 'pistachios'],
        'instructions': ['Hang yogurt overnight', 'Mix with sugar', 'Add saffron and cardamom', 'Beat smooth', 'Garnish and chill']
    },
    'indian_sohan_halwa': {
        'title': '🟤 Sohan Halwa',
        'cuisine': '🇮🇳 Indian (Multan)',
        'ingredients': ['wheat flour', 'ghee', 'sugar', 'milk', 'cardamom', 'almonds'],
        'instructions': ['Cook flour in ghee', 'Add milk gradually', 'Stir continuously', 'Add sugar', 'Set in tray']
    },
    'indian_sohan_papdi': {
        'title': '🍬 Sohan Papdi',
        'cuisine': '🇮🇳 Indian',
        'ingredients': ['flour', 'besan', 'ghee', 'sugar', 'cardamom', 'pistachios'],
        'instructions': ['Make sugar syrup', 'Cook flour and besan', 'Pull and stretch', 'Make flaky layers', 'Cut and serve']
    },
    'indian_sutar_feni': {
        'title': '🍝 Sutar Feni',
        'cuisine': '🇮🇳 Indian (Gujarat)',
        'ingredients': ['flour', 'ghee', 'sugar', 'saffron'],
        'instructions': ['Make dough', 'Pull into thin threads', 'Shape into rounds', 'Deep fry', 'Soak in syrup']
    },
    'indian_unni_appam': {
        'title': '🟤 Unni Appam',
        'cuisine': '🇮🇳 Indian (Kerala)',
        'ingredients': ['rice flour', 'banana', 'jaggery', 'coconut', 'cardamom', 'ghee'],
        'instructions': ['Mix ingredients', 'Make thick batter', 'Pour in appam pan', 'Fry both sides', 'Serve warm']
    },
    
    # ==================== WESTERN RECIPES (101) ====================
    'western_apple_pie': {
        'title': '🥧 Apple Pie',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['apples', 'pie crust', 'sugar', 'cinnamon', 'butter', 'lemon juice'],
        'instructions': ['Slice apples', 'Mix with sugar and spices', 'Fill pie crust', 'Add top crust', 'Bake at 375°F for 50 mins']
    },
    'western_baby_back_ribs': {
        'title': '🍖 Baby Back Ribs',
        'cuisine': '🍔 Western (American BBQ)',
        'ingredients': ['pork ribs', 'BBQ rub', 'BBQ sauce', 'apple cider vinegar', 'brown sugar'],
        'instructions': ['Remove membrane', 'Apply dry rub', 'Smoke or bake low and slow', 'Brush with sauce', 'Finish on high heat']
    },
    'western_baklava': {
        'title': '🍯 Baklava',
        'cuisine': '🍔 Western (Mediterranean)',
        'ingredients': ['phyllo dough', 'walnuts', 'butter', 'honey', 'cinnamon', 'sugar'],
        'instructions': ['Layer phyllo with butter', 'Add nut mixture', 'Cut into diamonds', 'Bake until golden', 'Pour honey syrup']
    },
    'western_beef_carpaccio': {
        'title': '🥩 Beef Carpaccio',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['beef tenderloin', 'arugula', 'parmesan', 'olive oil', 'lemon', 'capers'],
        'instructions': ['Freeze beef briefly', 'Slice paper thin', 'Arrange on plate', 'Top with arugula', 'Drizzle with oil and lemon']
    },
    'western_beef_tartare': {
        'title': '🥩 Beef Tartare',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['beef tenderloin', 'egg yolk', 'capers', 'shallots', 'dijon mustard', 'worcestershire'],
        'instructions': ['Hand chop beef', 'Mix with seasonings', 'Form into mound', 'Top with egg yolk', 'Serve with toast']
    },
    'western_beet_salad': {
        'title': '🥗 Beet Salad',
        'cuisine': '🍔 Western',
        'ingredients': ['beets', 'goat cheese', 'walnuts', 'arugula', 'balsamic vinaigrette'],
        'instructions': ['Roast beets', 'Cool and slice', 'Arrange on greens', 'Add cheese and nuts', 'Drizzle with dressing']
    },
    'western_beignets': {
        'title': '🍩 Beignets',
        'cuisine': '🍔 Western (New Orleans)',
        'ingredients': ['flour', 'yeast', 'milk', 'eggs', 'powdered sugar', 'oil'],
        'instructions': ['Make yeast dough', 'Let rise', 'Roll and cut squares', 'Deep fry', 'Dust with powdered sugar']
    },
    'western_bibimbap': {
        'title': '🍚 Bibimbap',
        'cuisine': '🍔 Western (Korean)',
        'ingredients': ['rice', 'beef', 'vegetables', 'gochujang', 'sesame oil', 'egg'],
        'instructions': ['Cook rice', 'Prepare vegetables', 'Cook beef', 'Arrange in bowl', 'Top with egg and gochujang']
    },
    'western_bread_pudding': {
        'title': '🍮 Bread Pudding',
        'cuisine': '🍔 Western (British)',
        'ingredients': ['bread', 'milk', 'eggs', 'sugar', 'vanilla', 'raisins', 'cinnamon'],
        'instructions': ['Cube bread', 'Make custard', 'Soak bread', 'Add raisins', 'Bake until set']
    },
    'western_breakfast_burrito': {
        'title': '🌯 Breakfast Burrito',
        'cuisine': '🍔 Western (Tex-Mex)',
        'ingredients': ['tortilla', 'eggs', 'bacon', 'cheese', 'salsa', 'potatoes'],
        'instructions': ['Scramble eggs', 'Cook bacon', 'Prepare potatoes', 'Wrap in tortilla', 'Add salsa and cheese']
    },
    'western_bruschetta': {
        'title': '🍞 Bruschetta',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['baguette', 'tomatoes', 'garlic', 'basil', 'olive oil', 'balsamic'],
        'instructions': ['Slice and toast bread', 'Dice tomatoes', 'Mix with garlic and basil', 'Top bread', 'Drizzle with oil']
    },
    'western_caesar_salad': {
        'title': '🥗 Caesar Salad',
        'cuisine': '🍔 Western',
        'ingredients': ['romaine lettuce', 'parmesan', 'croutons', 'caesar dressing', 'anchovies'],
        'instructions': ['Chop lettuce', 'Make dressing', 'Toss together', 'Add croutons', 'Top with parmesan']
    },
    'western_cannoli': {
        'title': '🥐 Cannoli',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['cannoli shells', 'ricotta', 'powdered sugar', 'chocolate chips', 'pistachios'],
        'instructions': ['Drain ricotta', 'Mix with sugar', 'Add chocolate chips', 'Fill shells', 'Dip ends in pistachios']
    },
    'western_caprese_salad': {
        'title': '🍅 Caprese Salad',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['tomatoes', 'fresh mozzarella', 'basil', 'olive oil', 'balsamic glaze'],
        'instructions': ['Slice tomatoes and mozzarella', 'Alternate on plate', 'Add basil leaves', 'Drizzle with oil', 'Add balsamic']
    },
    'western_carrot_cake': {
        'title': '🥕 Carrot Cake',
        'cuisine': '🍔 Western',
        'ingredients': ['carrots', 'flour', 'sugar', 'eggs', 'oil', 'cinnamon', 'cream cheese frosting'],
        'instructions': ['Grate carrots', 'Mix wet ingredients', 'Add dry ingredients', 'Bake', 'Frost when cool']
    },
    'western_ceviche': {
        'title': '🐟 Ceviche',
        'cuisine': '🍔 Western (Latin)',
        'ingredients': ['white fish', 'lime juice', 'red onion', 'cilantro', 'jalapeño', 'tomato'],
        'instructions': ['Dice fish', 'Marinate in lime juice', 'Add vegetables', 'Refrigerate 30 mins', 'Serve cold']
    },
    'western_cheese_plate': {
        'title': '🧀 Cheese Plate',
        'cuisine': '🍔 Western',
        'ingredients': ['assorted cheeses', 'crackers', 'grapes', 'nuts', 'honey', 'jam'],
        'instructions': ['Select variety of cheeses', 'Arrange on board', 'Add crackers', 'Garnish with fruits and nuts', 'Add honey']
    },
    'western_cheesecake': {
        'title': '🍰 Cheesecake',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['cream cheese', 'sugar', 'eggs', 'graham cracker crust', 'vanilla', 'sour cream'],
        'instructions': ['Make crust', 'Beat cream cheese', 'Add eggs one at a time', 'Pour into crust', 'Bake in water bath']
    },
    'western_chicken_curry': {
        'title': '🍛 Chicken Curry',
        'cuisine': '🍔 Western (British-Indian)',
        'ingredients': ['chicken', 'curry powder', 'coconut milk', 'onion', 'tomato', 'ginger'],
        'instructions': ['Brown chicken', 'Sauté onions', 'Add curry powder', 'Add coconut milk', 'Simmer until done']
    },
    'western_chicken_quesadilla': {
        'title': '🫔 Chicken Quesadilla',
        'cuisine': '🍔 Western (Tex-Mex)',
        'ingredients': ['tortillas', 'chicken', 'cheese', 'peppers', 'onion', 'sour cream'],
        'instructions': ['Cook chicken', 'Sauté vegetables', 'Layer in tortilla', 'Add cheese', 'Grill until crispy']
    },
    'western_chicken_wings': {
        'title': '🍗 Chicken Wings',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['chicken wings', 'butter', 'hot sauce', 'garlic powder', 'celery', 'blue cheese'],
        'instructions': ['Fry or bake wings', 'Make buffalo sauce', 'Toss wings in sauce', 'Serve with celery', 'Add blue cheese dip']
    },
    'western_chocolate_cake': {
        'title': '🍫 Chocolate Cake',
        'cuisine': '🍔 Western',
        'ingredients': ['flour', 'cocoa powder', 'sugar', 'eggs', 'butter', 'chocolate frosting'],
        'instructions': ['Mix dry ingredients', 'Add wet ingredients', 'Pour into pans', 'Bake 30 mins', 'Frost when cool']
    },
    'western_chocolate_mousse': {
        'title': '🍫 Chocolate Mousse',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['dark chocolate', 'eggs', 'cream', 'sugar', 'vanilla'],
        'instructions': ['Melt chocolate', 'Whip cream', 'Beat egg whites', 'Fold together', 'Chill 4 hours']
    },
    'western_churros': {
        'title': '🥖 Churros',
        'cuisine': '🍔 Western (Spanish)',
        'ingredients': ['flour', 'water', 'butter', 'sugar', 'cinnamon', 'chocolate sauce'],
        'instructions': ['Make choux dough', 'Pipe into hot oil', 'Fry until golden', 'Roll in cinnamon sugar', 'Serve with chocolate']
    },
    'western_clam_chowder': {
        'title': '🍲 Clam Chowder',
        'cuisine': '🍔 Western (New England)',
        'ingredients': ['clams', 'potatoes', 'cream', 'bacon', 'onion', 'celery'],
        'instructions': ['Cook bacon', 'Sauté vegetables', 'Add potatoes and broth', 'Add clams', 'Finish with cream']
    },
    'western_club_sandwich': {
        'title': '🥪 Club Sandwich',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['bread', 'turkey', 'bacon', 'lettuce', 'tomato', 'mayo'],
        'instructions': ['Toast bread', 'Cook bacon', 'Layer ingredients', 'Cut diagonally', 'Secure with picks']
    },
    'western_crab_cakes': {
        'title': '🦀 Crab Cakes',
        'cuisine': '🍔 Western (Maryland)',
        'ingredients': ['crab meat', 'breadcrumbs', 'egg', 'mayo', 'old bay', 'lemon'],
        'instructions': ['Mix ingredients gently', 'Form into cakes', 'Chill 30 mins', 'Pan fry', 'Serve with remoulade']
    },
    'western_creme_brulee': {
        'title': '🍮 Crème Brûlée',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['cream', 'egg yolks', 'sugar', 'vanilla bean'],
        'instructions': ['Heat cream with vanilla', 'Whisk yolks with sugar', 'Combine', 'Bake in water bath', 'Torch sugar top']
    },
    'western_croque_madame': {
        'title': '🥪 Croque Madame',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['bread', 'ham', 'gruyere', 'bechamel', 'egg', 'butter'],
        'instructions': ['Make bechamel', 'Assemble sandwich', 'Grill until golden', 'Top with more cheese', 'Add fried egg']
    },
    'western_cup_cakes': {
        'title': '🧁 Cupcakes',
        'cuisine': '🍔 Western',
        'ingredients': ['flour', 'sugar', 'butter', 'eggs', 'vanilla', 'frosting'],
        'instructions': ['Mix batter', 'Pour into liners', 'Bake 20 mins', 'Cool completely', 'Pipe frosting on top']
    },
    'western_deviled_eggs': {
        'title': '🥚 Deviled Eggs',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['eggs', 'mayo', 'mustard', 'paprika', 'chives'],
        'instructions': ['Hard boil eggs', 'Halve and remove yolks', 'Mix yolks with mayo', 'Pipe into whites', 'Garnish with paprika']
    },
    'western_donuts': {
        'title': '🍩 Donuts',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['flour', 'yeast', 'milk', 'sugar', 'butter', 'glaze'],
        'instructions': ['Make yeast dough', 'Let rise', 'Cut into rings', 'Fry until golden', 'Dip in glaze']
    },
    'western_dumplings': {
        'title': '🥟 Dumplings',
        'cuisine': '🍔 Western (Asian)',
        'ingredients': ['dumpling wrappers', 'pork', 'cabbage', 'ginger', 'soy sauce', 'sesame oil'],
        'instructions': ['Make filling', 'Wrap dumplings', 'Steam or pan fry', 'Make dipping sauce', 'Serve hot']
    },
    'western_edamame': {
        'title': '🫛 Edamame',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['edamame pods', 'sea salt', 'garlic', 'sesame oil'],
        'instructions': ['Boil edamame', 'Drain well', 'Toss with salt', 'Optional: add garlic', 'Serve warm or cold']
    },
    'western_eggs_benedict': {
        'title': '🍳 Eggs Benedict',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['english muffin', 'canadian bacon', 'eggs', 'hollandaise sauce', 'chives'],
        'instructions': ['Toast muffins', 'Warm bacon', 'Poach eggs', 'Make hollandaise', 'Assemble and sauce']
    },
    'western_escargots': {
        'title': '🐌 Escargots',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['snails', 'garlic butter', 'parsley', 'shallots', 'white wine'],
        'instructions': ['Prepare garlic butter', 'Place snails in shells', 'Top with butter', 'Bake until bubbling', 'Serve with bread']
    },
    'western_falafel': {
        'title': '🧆 Falafel',
        'cuisine': '🍔 Western (Middle Eastern)',
        'ingredients': ['chickpeas', 'herbs', 'garlic', 'cumin', 'tahini', 'pita'],
        'instructions': ['Soak chickpeas', 'Blend with herbs', 'Form balls', 'Deep fry', 'Serve in pita']
    },
    'western_filet_mignon': {
        'title': '🥩 Filet Mignon',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['beef tenderloin', 'butter', 'garlic', 'thyme', 'salt', 'pepper'],
        'instructions': ['Bring to room temp', 'Season generously', 'Sear in hot pan', 'Baste with butter', 'Rest before serving']
    },
    'western_fish_and_chips': {
        'title': '🐟 Fish and Chips',
        'cuisine': '🍔 Western (British)',
        'ingredients': ['cod', 'beer batter', 'potatoes', 'tartar sauce', 'malt vinegar', 'peas'],
        'instructions': ['Make batter', 'Cut chips', 'Fry chips', 'Batter and fry fish', 'Serve with tartar sauce']
    },
    'western_foie_gras': {
        'title': '🍖 Foie Gras',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['duck liver', 'brioche', 'fig jam', 'sea salt', 'white pepper'],
        'instructions': ['Slice foie gras', 'Sear briefly', 'Toast brioche', 'Add fig jam', 'Serve immediately']
    },
    'western_french_fries': {
        'title': '🍟 French Fries',
        'cuisine': '🍔 Western',
        'ingredients': ['potatoes', 'oil', 'salt', 'ketchup'],
        'instructions': ['Cut into strips', 'Soak in water', 'Dry thoroughly', 'Double fry', 'Season with salt']
    },
    'western_french_onion_soup': {
        'title': '🍲 French Onion Soup',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['onions', 'beef broth', 'bread', 'gruyere', 'butter', 'thyme'],
        'instructions': ['Caramelize onions slowly', 'Add broth', 'Simmer', 'Ladle into bowls', 'Top with bread and cheese', 'Broil']
    },
    'western_french_toast': {
        'title': '🍞 French Toast',
        'cuisine': '🍔 Western',
        'ingredients': ['bread', 'eggs', 'milk', 'cinnamon', 'maple syrup', 'butter'],
        'instructions': ['Make egg mixture', 'Dip bread', 'Cook in butter', 'Flip when golden', 'Serve with syrup']
    },
    'western_fried_calamari': {
        'title': '🦑 Fried Calamari',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['squid', 'flour', 'cornmeal', 'marinara sauce', 'lemon', 'parsley'],
        'instructions': ['Clean squid', 'Cut into rings', 'Dredge in flour', 'Deep fry', 'Serve with marinara']
    },
    'western_fried_rice': {
        'title': '🍚 Fried Rice',
        'cuisine': '🍔 Western (Chinese)',
        'ingredients': ['day-old rice', 'eggs', 'vegetables', 'soy sauce', 'sesame oil', 'green onions'],
        'instructions': ['Heat wok', 'Scramble eggs', 'Add rice', 'Add vegetables', 'Season with soy sauce']
    },
    'western_frozen_yogurt': {
        'title': '🍦 Frozen Yogurt',
        'cuisine': '🍔 Western',
        'ingredients': ['yogurt', 'sugar', 'vanilla', 'fruits', 'toppings'],
        'instructions': ['Mix yogurt and sugar', 'Add vanilla', 'Churn in ice cream maker', 'Freeze', 'Add toppings']
    },
    'western_garlic_bread': {
        'title': '🍞 Garlic Bread',
        'cuisine': '🍔 Western (Italian-American)',
        'ingredients': ['baguette', 'butter', 'garlic', 'parsley', 'parmesan'],
        'instructions': ['Make garlic butter', 'Slice bread', 'Spread butter', 'Add cheese', 'Bake until golden']
    },
    'western_gnocchi': {
        'title': '🥔 Gnocchi',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['potatoes', 'flour', 'egg', 'nutmeg', 'sage butter', 'parmesan'],
        'instructions': ['Bake potatoes', 'Rice while warm', 'Add flour and egg', 'Shape gnocchi', 'Boil and sauce']
    },
    'western_greek_salad': {
        'title': '🥗 Greek Salad',
        'cuisine': '🍔 Western (Greek)',
        'ingredients': ['cucumber', 'tomato', 'feta', 'olives', 'red onion', 'olive oil', 'oregano'],
        'instructions': ['Chop vegetables', 'Add olives and feta', 'Drizzle with oil', 'Season with oregano', 'Toss gently']
    },
    'western_grilled_cheese_sandwich': {
        'title': '🥪 Grilled Cheese',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['bread', 'cheese', 'butter'],
        'instructions': ['Butter bread', 'Add cheese', 'Grill low and slow', 'Flip when golden', 'Serve hot']
    },
    'western_grilled_salmon': {
        'title': '🐟 Grilled Salmon',
        'cuisine': '🍔 Western',
        'ingredients': ['salmon fillet', 'olive oil', 'lemon', 'dill', 'garlic', 'salt'],
        'instructions': ['Season salmon', 'Heat grill', 'Grill skin-side down', 'Flip once', 'Serve with lemon']
    },
    'western_guacamole': {
        'title': '🥑 Guacamole',
        'cuisine': '🍔 Western (Mexican)',
        'ingredients': ['avocados', 'lime', 'cilantro', 'onion', 'jalapeño', 'tomato'],
        'instructions': ['Mash avocados', 'Add lime juice', 'Mix in onion', 'Add cilantro', 'Season to taste']
    },
    'western_gyoza': {
        'title': '🥟 Gyoza',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['gyoza wrappers', 'pork', 'cabbage', 'garlic', 'ginger', 'soy sauce'],
        'instructions': ['Make filling', 'Wrap dumplings', 'Pan fry bottoms', 'Add water and steam', 'Serve with dipping sauce']
    },
    'western_hamburger': {
        'title': '🍔 Hamburger',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['ground beef', 'burger buns', 'lettuce', 'tomato', 'onion', 'cheese', 'pickles'],
        'instructions': ['Form patties', 'Season with salt and pepper', 'Grill to preference', 'Toast buns', 'Assemble with toppings']
    },
    'western_hot_and_sour_soup': {
        'title': '🍲 Hot and Sour Soup',
        'cuisine': '🍔 Western (Chinese)',
        'ingredients': ['tofu', 'mushrooms', 'bamboo shoots', 'egg', 'vinegar', 'white pepper'],
        'instructions': ['Make broth', 'Add vegetables', 'Season with vinegar', 'Add egg ribbons', 'Thicken with cornstarch']
    },
    'western_hot_dog': {
        'title': '🌭 Hot Dog',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['hot dogs', 'buns', 'mustard', 'ketchup', 'relish', 'onions'],
        'instructions': ['Grill or boil hot dogs', 'Toast buns', 'Place in bun', 'Add condiments', 'Serve immediately']
    },
    'western_huevos_rancheros': {
        'title': '🍳 Huevos Rancheros',
        'cuisine': '🍔 Western (Mexican)',
        'ingredients': ['eggs', 'tortillas', 'salsa roja', 'beans', 'cheese', 'avocado'],
        'instructions': ['Fry tortillas', 'Fry eggs', 'Warm salsa', 'Assemble', 'Top with cheese']
    },
    'western_hummus': {
        'title': '🥣 Hummus',
        'cuisine': '🍔 Western (Middle Eastern)',
        'ingredients': ['chickpeas', 'tahini', 'lemon', 'garlic', 'olive oil', 'cumin'],
        'instructions': ['Blend chickpeas', 'Add tahini', 'Add lemon and garlic', 'Drizzle with oil', 'Serve with pita']
    },
    'western_ice_cream': {
        'title': '🍨 Ice Cream',
        'cuisine': '🍔 Western',
        'ingredients': ['cream', 'milk', 'sugar', 'egg yolks', 'vanilla'],
        'instructions': ['Make custard base', 'Cool completely', 'Churn in machine', 'Freeze', 'Serve with toppings']
    },
    'western_lasagna': {
        'title': '🍝 Lasagna',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['lasagna noodles', 'meat sauce', 'ricotta', 'mozzarella', 'parmesan', 'bechamel'],
        'instructions': ['Make meat sauce', 'Layer noodles and cheese', 'Repeat layers', 'Top with mozzarella', 'Bake 45 mins']
    },
    'western_lobster_bisque': {
        'title': '🦞 Lobster Bisque',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['lobster', 'cream', 'sherry', 'tomato paste', 'onion', 'celery'],
        'instructions': ['Cook lobster', 'Make shell stock', 'Sauté vegetables', 'Add cream', 'Garnish with lobster']
    },
    'western_lobster_roll_sandwich': {
        'title': '🦞 Lobster Roll',
        'cuisine': '🍔 Western (New England)',
        'ingredients': ['lobster meat', 'hot dog buns', 'mayo', 'lemon', 'celery', 'butter'],
        'instructions': ['Cook lobster', 'Mix with mayo', 'Toast buttered buns', 'Fill with lobster', 'Serve cold']
    },
    'western_macaroni_and_cheese': {
        'title': '🧀 Mac and Cheese',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['macaroni', 'cheddar', 'milk', 'butter', 'flour', 'breadcrumbs'],
        'instructions': ['Cook pasta', 'Make cheese sauce', 'Combine', 'Top with breadcrumbs', 'Bake until bubbly']
    },
    'western_macarons': {
        'title': '🍪 Macarons',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['almond flour', 'powdered sugar', 'egg whites', 'sugar', 'food coloring', 'filling'],
        'instructions': ['Make meringue', 'Fold in almond flour', 'Pipe circles', 'Rest then bake', 'Fill with ganache']
    },
    'western_miso_soup': {
        'title': '🍜 Miso Soup',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['dashi', 'miso paste', 'tofu', 'wakame', 'green onions'],
        'instructions': ['Heat dashi', 'Dissolve miso', 'Add tofu', 'Add wakame', 'Garnish with green onions']
    },
    'western_mussels': {
        'title': '🦪 Mussels',
        'cuisine': '🍔 Western (Belgian/French)',
        'ingredients': ['mussels', 'white wine', 'garlic', 'shallots', 'cream', 'parsley'],
        'instructions': ['Clean mussels', 'Sauté shallots and garlic', 'Add wine', 'Add mussels and cover', 'Serve with bread']
    },
    'western_nachos': {
        'title': '🌮 Nachos',
        'cuisine': '🍔 Western (Tex-Mex)',
        'ingredients': ['tortilla chips', 'cheese', 'jalapeños', 'beans', 'sour cream', 'guacamole'],
        'instructions': ['Layer chips on pan', 'Add cheese and toppings', 'Bake until melted', 'Add cold toppings', 'Serve immediately']
    },
    'western_omelette': {
        'title': '🍳 Omelette',
        'cuisine': '🍔 Western (French)',
        'ingredients': ['eggs', 'butter', 'cheese', 'herbs', 'vegetables'],
        'instructions': ['Beat eggs', 'Cook in butter', 'Add fillings', 'Fold over', 'Serve immediately']
    },
    'western_onion_rings': {
        'title': '🧅 Onion Rings',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['onions', 'flour', 'buttermilk', 'breadcrumbs', 'oil'],
        'instructions': ['Slice onions thick', 'Dip in buttermilk', 'Coat in breading', 'Deep fry', 'Season with salt']
    },
    'western_oysters': {
        'title': '🦪 Oysters',
        'cuisine': '🍔 Western',
        'ingredients': ['fresh oysters', 'lemon', 'mignonette', 'cocktail sauce', 'horseradish'],
        'instructions': ['Shuck oysters', 'Place on ice', 'Make mignonette', 'Arrange sauces', 'Serve immediately']
    },
    'western_pad_thai': {
        'title': '🍜 Pad Thai',
        'cuisine': '🍔 Western (Thai)',
        'ingredients': ['rice noodles', 'shrimp', 'eggs', 'bean sprouts', 'peanuts', 'tamarind sauce'],
        'instructions': ['Soak noodles', 'Stir fry protein', 'Add eggs', 'Add noodles and sauce', 'Top with peanuts']
    },
    'western_paella': {
        'title': '🥘 Paella',
        'cuisine': '🍔 Western (Spanish)',
        'ingredients': ['rice', 'saffron', 'chicken', 'seafood', 'chorizo', 'bell peppers'],
        'instructions': ['Make sofrito', 'Add rice and saffron', 'Add broth', 'Add proteins', 'Cook without stirring']
    },
    'western_pancakes': {
        'title': '🥞 Pancakes',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['flour', 'eggs', 'milk', 'butter', 'maple syrup', 'baking powder'],
        'instructions': ['Mix batter', 'Rest 5 mins', 'Cook on griddle', 'Flip when bubbly', 'Serve with syrup']
    },
    'western_panna_cotta': {
        'title': '🍮 Panna Cotta',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['cream', 'sugar', 'gelatin', 'vanilla', 'berry sauce'],
        'instructions': ['Bloom gelatin', 'Heat cream and sugar', 'Add gelatin', 'Pour into molds', 'Chill until set']
    },
    'western_peking_duck': {
        'title': '🦆 Peking Duck',
        'cuisine': '🍔 Western (Chinese)',
        'ingredients': ['whole duck', 'maltose', 'soy sauce', 'pancakes', 'hoisin', 'scallions'],
        'instructions': ['Air dry duck', 'Glaze with maltose', 'Roast until crispy', 'Slice', 'Serve with pancakes']
    },
    'western_pho': {
        'title': '🍜 Pho',
        'cuisine': '🍔 Western (Vietnamese)',
        'ingredients': ['rice noodles', 'beef broth', 'beef', 'star anise', 'herbs', 'bean sprouts'],
        'instructions': ['Simmer broth with spices', 'Cook noodles', 'Slice beef thin', 'Assemble bowls', 'Serve with herbs']
    },
    'western_pizza': {
        'title': '🍕 Pizza',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['pizza dough', 'tomato sauce', 'mozzarella', 'basil', 'olive oil'],
        'instructions': ['Stretch dough', 'Add sauce', 'Top with cheese', 'Add toppings', 'Bake at high heat']
    },
    'western_pork_chop': {
        'title': '🥩 Pork Chop',
        'cuisine': '🍔 Western',
        'ingredients': ['pork chops', 'garlic', 'rosemary', 'butter', 'apple sauce'],
        'instructions': ['Season chops', 'Sear in hot pan', 'Add butter and herbs', 'Baste', 'Rest before serving']
    },
    'western_poutine': {
        'title': '🍟 Poutine',
        'cuisine': '🍔 Western (Canadian)',
        'ingredients': ['french fries', 'cheese curds', 'gravy'],
        'instructions': ['Make crispy fries', 'Heat gravy', 'Layer fries and curds', 'Pour hot gravy', 'Serve immediately']
    },
    'western_prime_rib': {
        'title': '🥩 Prime Rib',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['beef rib roast', 'garlic', 'herbs', 'butter', 'au jus', 'horseradish'],
        'instructions': ['Season roast', 'Roast at high then low', 'Rest 30 mins', 'Make au jus', 'Slice and serve']
    },
    'western_pulled_pork_sandwich': {
        'title': '🥪 Pulled Pork',
        'cuisine': '🍔 Western (American BBQ)',
        'ingredients': ['pork shoulder', 'BBQ rub', 'BBQ sauce', 'buns', 'coleslaw'],
        'instructions': ['Apply rub', 'Smoke or slow cook', 'Shred meat', 'Mix with sauce', 'Serve on buns']
    },
    'western_ramen': {
        'title': '🍜 Ramen',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['ramen noodles', 'pork broth', 'chashu', 'soft egg', 'nori', 'green onions'],
        'instructions': ['Make rich broth', 'Cook noodles', 'Prepare toppings', 'Assemble bowl', 'Serve hot']
    },
    'western_ravioli': {
        'title': '🥟 Ravioli',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['pasta dough', 'ricotta', 'spinach', 'parmesan', 'sage butter'],
        'instructions': ['Make pasta dough', 'Make filling', 'Fill and seal', 'Boil gently', 'Toss in sage butter']
    },
    'western_red_velvet_cake': {
        'title': '🍰 Red Velvet Cake',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['flour', 'cocoa', 'red food coloring', 'buttermilk', 'cream cheese frosting'],
        'instructions': ['Mix wet ingredients', 'Add dry ingredients', 'Bake in layers', 'Cool completely', 'Frost with cream cheese']
    },
    'western_risotto': {
        'title': '🍚 Risotto',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['arborio rice', 'broth', 'white wine', 'parmesan', 'butter', 'onion'],
        'instructions': ['Toast rice', 'Add wine', 'Add broth gradually', 'Stir constantly', 'Finish with butter and cheese']
    },
    'western_samosa': {
        'title': '🥟 Samosa',
        'cuisine': '🍔 Western (Indian)',
        'ingredients': ['flour', 'potatoes', 'peas', 'cumin', 'coriander', 'green chili'],
        'instructions': ['Make dough', 'Prepare filling', 'Shape into triangles', 'Deep fry', 'Serve with chutney']
    },
    'western_sashimi': {
        'title': '🍣 Sashimi',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['fresh fish', 'wasabi', 'soy sauce', 'pickled ginger', 'daikon'],
        'instructions': ['Select sushi-grade fish', 'Slice against grain', 'Arrange beautifully', 'Serve with wasabi', 'Dip in soy sauce']
    },
    'western_scallops': {
        'title': '🦪 Scallops',
        'cuisine': '🍔 Western',
        'ingredients': ['sea scallops', 'butter', 'garlic', 'lemon', 'parsley'],
        'instructions': ['Pat scallops dry', 'Season with salt', 'Sear in hot pan', 'Baste with butter', 'Serve immediately']
    },
    'western_seaweed_salad': {
        'title': '🥗 Seaweed Salad',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['wakame', 'sesame oil', 'rice vinegar', 'soy sauce', 'sesame seeds'],
        'instructions': ['Rehydrate seaweed', 'Make dressing', 'Toss together', 'Garnish with sesame', 'Chill and serve']
    },
    'western_shrimp_and_grits': {
        'title': '🦐 Shrimp and Grits',
        'cuisine': '🍔 Western (Southern)',
        'ingredients': ['shrimp', 'grits', 'bacon', 'cheese', 'garlic', 'green onions'],
        'instructions': ['Cook creamy grits', 'Cook bacon', 'Sauté shrimp', 'Make pan sauce', 'Serve over grits']
    },
    'western_spaghetti_bolognese': {
        'title': '🍝 Spaghetti Bolognese',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['spaghetti', 'ground beef', 'tomatoes', 'onion', 'carrots', 'celery', 'red wine'],
        'instructions': ['Make soffritto', 'Brown meat', 'Add tomatoes', 'Simmer 2 hours', 'Serve over pasta']
    },
    'western_spaghetti_carbonara': {
        'title': '🍝 Spaghetti Carbonara',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['spaghetti', 'guanciale', 'egg yolks', 'pecorino', 'black pepper'],
        'instructions': ['Cook pasta', 'Crisp guanciale', 'Mix eggs and cheese', 'Toss hot pasta', 'Add pepper']
    },
    'western_spring_rolls': {
        'title': '🥢 Spring Rolls',
        'cuisine': '🍔 Western (Asian)',
        'ingredients': ['spring roll wrappers', 'vegetables', 'vermicelli', 'shrimp', 'sweet chili sauce'],
        'instructions': ['Prepare filling', 'Wrap tightly', 'Deep fry or serve fresh', 'Make dipping sauce', 'Serve hot or cold']
    },
    'western_steak': {
        'title': '🥩 Steak',
        'cuisine': '🍔 Western',
        'ingredients': ['ribeye/sirloin', 'butter', 'garlic', 'thyme', 'salt', 'pepper'],
        'instructions': ['Bring to room temp', 'Season generously', 'Sear in hot pan', 'Baste with butter', 'Rest 5 mins']
    },
    'western_strawberry_shortcake': {
        'title': '🍓 Strawberry Shortcake',
        'cuisine': '🍔 Western (American)',
        'ingredients': ['biscuits', 'strawberries', 'whipped cream', 'sugar', 'vanilla'],
        'instructions': ['Bake biscuits', 'Macerate strawberries', 'Whip cream', 'Split biscuits', 'Layer and serve']
    },
    'western_sushi': {
        'title': '🍣 Sushi',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['sushi rice', 'nori', 'fish', 'cucumber', 'wasabi', 'soy sauce'],
        'instructions': ['Season rice', 'Prepare fish', 'Roll with nori', 'Slice into pieces', 'Serve with wasabi']
    },
    'western_tacos': {
        'title': '🌮 Tacos',
        'cuisine': '🍔 Western (Mexican)',
        'ingredients': ['tortillas', 'meat', 'onion', 'cilantro', 'salsa', 'lime'],
        'instructions': ['Season and cook meat', 'Warm tortillas', 'Add meat', 'Top with onion and cilantro', 'Squeeze lime']
    },
    'western_takoyaki': {
        'title': '🐙 Takoyaki',
        'cuisine': '🍔 Western (Japanese)',
        'ingredients': ['batter', 'octopus', 'green onion', 'takoyaki sauce', 'mayo', 'bonito flakes'],
        'instructions': ['Make batter', 'Pour into molds', 'Add octopus', 'Turn continuously', 'Top with sauce']
    },
    'western_tiramisu': {
        'title': '🍰 Tiramisu',
        'cuisine': '🍔 Western (Italian)',
        'ingredients': ['ladyfingers', 'mascarpone', 'espresso', 'cocoa', 'egg yolks', 'marsala'],
        'instructions': ['Make mascarpone cream', 'Dip ladyfingers in coffee', 'Layer cream and cookies', 'Refrigerate overnight', 'Dust with cocoa']
    },
    'western_tuna_tartare': {
        'title': '🐟 Tuna Tartare',
        'cuisine': '🍔 Western',
        'ingredients': ['sushi-grade tuna', 'soy sauce', 'sesame oil', 'avocado', 'sriracha', 'wonton chips'],
        'instructions': ['Dice tuna finely', 'Mix with seasonings', 'Add avocado', 'Serve on chips', 'Garnish with sesame']
    },
    'western_waffles': {
        'title': '🧇 Waffles',
        'cuisine': '🍔 Western (Belgian)',
        'ingredients': ['flour', 'eggs', 'milk', 'butter', 'sugar', 'maple syrup'],
        'instructions': ['Mix batter', 'Preheat waffle iron', 'Cook until golden', 'Serve immediately', 'Top with syrup']
    },
}

# Default recipes
DEFAULT_INDIAN = {
    'title': '🍛 Indian Dish',
    'cuisine': '🇮🇳 Indian',
    'ingredients': ['spices', 'vegetables', 'oil/ghee', 'onion', 'tomato', 'ginger-garlic'],
    'instructions': ['Prepare ingredients', 'Temper spices in oil', 'Add aromatics', 'Cook main ingredients', 'Garnish and serve']
}

DEFAULT_WESTERN = {
    'title': '🍽️ Western Dish',
    'cuisine': '🍔 Western',
    'ingredients': ['protein', 'vegetables', 'seasonings', 'butter/oil', 'herbs'],
    'instructions': ['Prep all ingredients', 'Season the protein', 'Cook to desired doneness', 'Prepare sides', 'Plate and serve']
}

def get_recipe(class_name):
    """Get recipe based on predicted class"""
    if class_name in RECIPES:
        return RECIPES[class_name]
    
    # Try partial match
    for key in RECIPES:
        key_base = key.replace('indian_', '').replace('western_', '')
        class_base = class_name.replace('indian_', '').replace('western_', '')
        if key_base in class_base or class_base in key_base:
            return RECIPES[key]
    
    # Return default
    if class_name.startswith('indian_'):
        recipe = DEFAULT_INDIAN.copy()
        recipe['title'] = f"🍛 {class_name.replace('indian_', '').replace('_', ' ').title()}"
    else:
        recipe = DEFAULT_WESTERN.copy()
        recipe['title'] = f"🍽️ {class_name.replace('western_', '').replace('_', ' ').title()}"
    
    return recipe

# =============================================================================
# LOAD MODEL
# =============================================================================
MODEL = None
CLASS_TO_IDX = None

def load_model():
    global MODEL, CLASS_TO_IDX
    
    model_path = 'model/best_model.pth'
    
    if not os.path.exists(model_path):
        return False, "Model not found!"
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    CLASS_TO_IDX = checkpoint['class_to_idx']
    
    MODEL = LargeFoodClassifier(num_classes=len(CLASS_TO_IDX))
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.to(device)
    MODEL.eval()
    
    indian_count = sum(1 for c in CLASS_TO_IDX if c.startswith('indian_'))
    western_count = sum(1 for c in CLASS_TO_IDX if c.startswith('western_'))
    accuracy = checkpoint.get('val_acc', 84.8)
    
    print(f"✅ Model loaded! {indian_count} Indian + {western_count} Western = {len(CLASS_TO_IDX)} categories")
    print(f"📊 Model accuracy: {accuracy:.1f}%")
    return True, f"Loaded: {indian_count} Indian + {western_count} Western ({accuracy:.1f}% accuracy)"

# =============================================================================
# PREDICTION
# =============================================================================
def predict_food(image):
    if MODEL is None:
        success, msg = load_model()
        if not success:
            return msg, "", "", ""
    
    if image is None:
        return "Please upload an image!", "", "", ""
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert('RGB')
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = MODEL(tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_idx = probs.topk(5)
    
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    
    top_class = idx_to_class.get(top_idx[0][0].item(), 'unknown')
    top_prob = top_probs[0][0].item()
    
    cuisine_emoji = "🇮🇳" if top_class.startswith('indian_') else "🍔"
    cuisine_name = "Indian" if top_class.startswith('indian_') else "Western"
    display_name = top_class.replace('indian_', '').replace('western_', '').replace('_', ' ').title()
    
    prediction_text = f"## {cuisine_emoji} **{display_name}**\n"
    prediction_text += f"### Cuisine: {cuisine_name} | Confidence: {top_prob*100:.1f}%\n\n"
    prediction_text += "### 📊 Top 5 Predictions:\n"
    
    for i in range(5):
        cls = idx_to_class.get(top_idx[0][i].item(), 'unknown')
        prob = top_probs[0][i].item()
        emoji = "🇮🇳" if cls.startswith('indian_') else "🍔"
        name = cls.replace('indian_', '').replace('western_', '').replace('_', ' ').title()
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        prediction_text += f"{i+1}. {emoji} **{name}** {bar} {prob*100:.1f}%\n"
    
    recipe = get_recipe(top_class)
    
    recipe_title = f"# {recipe['title']}\n### {recipe['cuisine']}"
    
    ingredients_text = "## 🥘 Ingredients\n"
    for ing in recipe['ingredients']:
        ingredients_text += f"- {ing}\n"
    
    instructions_text = "## 👨‍🍳 Instructions\n"
    for i, step in enumerate(recipe['instructions'], 1):
        instructions_text += f"**{i}.** {step}\n\n"
    
    return prediction_text, recipe_title, ingredients_text, instructions_text

# =============================================================================
# GRADIO INTERFACE
# =============================================================================
print("\n📦 Loading model...")
load_model()

with gr.Blocks(title="Food Recipe Generator - 181 Categories", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🍛🍔 Universal Food Recipe Generator
    ### Recognizes **181 food categories** with **84.8% accuracy**!
    
    **80 Indian dishes** + **101 International dishes**
    
    *Upload any food photo and get the recipe instantly*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="📷 Upload Food Image", type="pil", height=350)
            submit_btn = gr.Button("🔍 Identify Food & Get Recipe", variant="primary", size="lg")
            
            with gr.Accordion("📋 Supported Foods", open=False):
                gr.Markdown("""
                ### 🇮🇳 Indian Foods (80):
                Biryani, Butter Chicken, Naan, Dosa, Paneer Tikka, Dal Makhani, 
                Gulab Jamun, Jalebi, Rasgulla, Palak Paneer, Chole, Samosa, Idli...
                
                ### 🍔 International Foods (101):
                Pizza, Hamburger, Sushi, Ramen, Pad Thai, Tacos, Pasta, Steak,
                Cheesecake, Tiramisu, Caesar Salad, Fish & Chips, Paella...
                """)
        
        with gr.Column(scale=1):
            prediction_output = gr.Markdown(label="Prediction")
            recipe_title = gr.Markdown(label="Recipe")
    
    with gr.Row():
        with gr.Column():
            ingredients_output = gr.Markdown(label="Ingredients")
        with gr.Column():
            instructions_output = gr.Markdown(label="Instructions")
    
    submit_btn.click(
        fn=predict_food,
        inputs=image_input,
        outputs=[prediction_output, recipe_title, ingredients_output, instructions_output]
    )
    
    gr.Markdown("""
    ---
    **Model:** EfficientNet-B0 | **Accuracy:** 84.8% | **Categories:** 181 (80 Indian + 101 Western) | **Training Gap:** +2.5% (No Overfitting!)
    """)

if __name__ == "__main__":
    print("\n🚀 Starting server at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
