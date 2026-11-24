import json, re, unicodedata
from pathlib import Path
import numpy as np
import torch
from torch import nn
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / 'recipe_model1' / 'ingredient_set_encoder.pt'
KG_PATH = BASE_DIR / 'recipe_model1' / 'knowledge_graph.json'
RAW_DATA_PATH = BASE_DIR / 'data' / 'dish_knowledge_base_with_images.json'
if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = BASE_DIR / 'dish_knowledge_base_with_images.json'

# Load detailed recipe instructions and steps
dish_details = {}
if RAW_DATA_PATH.exists():
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        raw_dishes = json.load(f)
        dish_details = {d.get('id'): d for d in raw_dishes}

# Global model state
print("Đang nạp mô hình Deep Hybrid...")
ckpt = torch.load(MODEL_PATH, map_location='cpu')
vocab = ckpt['vocab']
synonym_to_canonical = ckpt.get('synonym_to_canonical', {})
default_pantry = set(ckpt.get('default_pantry', []))

# Bổ sung các biến thể gia vị phổ biến vào từ điển đồng nghĩa & tủ bếp mặc định
SPICE_EXPANSIONS = {
    # Tiêu và các biến thể
    'tieu xay': 'tieu', 'tieu bot': 'tieu', 'tieu den': 'tieu', 'tieu den xay': 'tieu',
    'hat tieu': 'tieu', 'hat tieu xay': 'tieu', 'tieu hat': 'tieu', 'tieu trang': 'tieu',
    'tieu trang xay': 'tieu', 'tieu so': 'tieu', 'tieu so xay': 'tieu',
    # Muối
    'muoi tinh': 'muoi', 'muoi hat': 'muoi', 'muoi iot': 'muoi', 'muoi bot': 'muoi', 'muoi an': 'muoi',
    # Đường
    'duong cat': 'duong', 'duong trang': 'duong', 'duong cat trang': 'duong', 'duong tinh luyen': 'duong',
    # Tỏi
    'toi bam': 'toi', 'cu toi': 'toi', 'toi phi': 'toi',
    # Hành
    'hanh kho': 'hanh tim', 'hanh cu': 'hanh tim', 'hanh tim bam': 'hanh tim',
    # Nước mắm, dầu ăn
    'nuoc mam ngon': 'nuoc mam', 'dau thuc vat': 'dau an',
    # Ớt
    'ot hiem': 'ot', 'ot trai': 'ot', 'ot sung': 'ot',
    # Bột ngọt / Mì chính
    'mi chinh': 'bot ngot'
}
synonym_to_canonical.update(SPICE_EXPANSIONS)
default_pantry.update(SPICE_EXPANSIONS.keys())

best_coeff = ckpt.get('best_coeff', (0.25, 0.25, 0.5, 0.0))
recipes = ckpt['recipes']
names = ckpt['names']
keys = [None] + sorted(names)

with open(KG_PATH, 'r', encoding='utf-8') as f:
    kg_data = json.load(f)
kg = kg_data.get('kg', {})
categories = kg_data.get('categories', [])

# Reconstruct weights matrix
weights = np.zeros((len(recipes), len(vocab)+1), dtype=np.float32)
for i, dish in enumerate(recipes):
    for key, w in dish['ingredients'].items():
        if key in vocab:
            weights[i, vocab[key]] = w
row_sums = np.maximum(weights.sum(axis=1), 1e-8)
recipe_ids = [np.flatnonzero(row).tolist() for row in weights]

# Inverse KG for fast candidate lookup
inv_kg = {}
for did, edges in kg.items():
    for edge in edges:
        target = edge.get('target')
        inv_kg.setdefault(target, []).append({'did': did, 'rel': edge.get('rel'), 'importance': edge.get('importance')})

recipe_idx_by_id = {d['id']: i for i, d in enumerate(recipes)}

# DeepSets Encoder Architecture
class IngredientSetEncoder(nn.Module):
    def __init__(self, n_ing, n_recipes, dim=96, hidden=192, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(n_ing + 1, dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_recipes)
        )
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        pooled = (self.embedding(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.net(pooled)

model = IngredientSetEncoder(len(vocab), len(recipes))
model.load_state_dict(ckpt['state_dict'])
model.eval()
print(f"Mô hình đã sẵn sàng với {len(recipes)} công thức và {len(vocab)} nguyên liệu chuẩn hóa!")

def normalize(value):
    s = unicodedata.normalize('NFKC', str(value or '')).lower().strip().replace('\u0111', 'd')
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', s)).strip()

def resolve_ingredient(name):
    norm = normalize(name)
    return synonym_to_canonical.get(norm, norm)

def features_batch(x):
    present = np.zeros((len(x), len(vocab)+1), dtype=np.float32)
    for i, row in enumerate(x.numpy()):
        present[i, row[row != 0]] = 1
    matched = present @ (weights > 0).astype(np.float32).T
    coverage = matched / np.maximum(present.sum(axis=1, keepdims=True), 1)
    cbf = (present @ weights.T) / row_sums[None, :]
    main = ((weights > 0) & (weights == weights.max(axis=1, keepdims=True))).astype(np.float32)
    main_count = main.sum(axis=1)
    main_score = np.divide(present @ main.T, main_count[None, :],
                           out=np.ones((len(x), len(recipes)), dtype=np.float32),
                           where=main_count[None, :] > 0)
    with torch.no_grad():
        logits = model(x).numpy()
    lo, hi = logits.min(axis=1, keepdims=True), logits.max(axis=1, keepdims=True)
    deep = (logits - lo) / np.maximum(hi - lo, 1e-8)
    return deep, cbf, coverage, main_score, matched

def graph_candidates(ingredient_ids, category=None):
    candidates = set()
    for iid in ingredient_ids:
        k = keys[iid]
        for edge in inv_kg.get(k, []):
            did = edge.get('did')
            if did in recipe_idx_by_id:
                candidates.add(recipe_idx_by_id[did])
    if category:
        cat_matches = {recipe_idx_by_id[edge['did']] for edge in inv_kg.get(category, []) if edge.get('did') in recipe_idx_by_id}
        candidates &= cat_matches
    return candidates

def graph_explanation(dish_idx, user_ing_ids):
    dish = recipes[dish_idx]
    did = dish['id']
    user_keys = {keys[i] for i in user_ing_ids}
    matched, missing = [], []
    for edge in kg.get(did, []):
        if edge.get('rel') == 'requires':
            name = edge.get('target')
            item = {'name': names.get(name, name), 'importance': edge.get('importance', 1)}
            if name in user_keys:
                matched.append(item)
            else:
                missing.append(item)
    return {'matched': matched, 'missing': missing, 'category': dish.get('category')}

def perform_recommend(ingredients, top_k=10, method='hybrid', category=None, assume_pantry=True):
    if isinstance(ingredients, str):
        raw_items = [s.strip() for s in ingredients.split(',') if s.strip()]
    else:
        raw_items = [str(s).strip() for s in ingredients if str(s).strip()]
        
    resolved = [resolve_ingredient(s) for s in raw_items]
    unknown = [raw_items[i] for i, k in enumerate(resolved) if k not in vocab]
    ids = sorted({vocab[k] for k in resolved if k in vocab})
    
    optional = set(default_pantry) if assume_pantry else set()
    result = {'ready': [], 'needs_shopping': [], 'unknown': unknown, 'query_resolved': [names.get(keys[i], keys[i]) for i in ids]}
    if not ids:
        return result
        
    x = torch.tensor([ids])
    deep, cbf, coverage, main, matched = features_batch(x)
    
    configs = {
        'hybrid': best_coeff,
        'cbf': (0, 1, 0, 0),
        'encoder': (1, 0, 0, 0),
        'main_aware': (0.25, 0.25, 0.25, 0.25)
    }
    coeff = configs.get(method, best_coeff)
    scores = sum(w * f[0] for w, f in zip(coeff, (deep, cbf, coverage, main)))
    
    candidates_set = graph_candidates(ids, category=category if category != 'all' else None)
    
    for idx in np.argsort(-scores):
        if idx not in candidates_set:
            continue
        dish = recipes[idx]
        have = set(ids)
        missing_keys = [k for k in dish['ingredients'] if vocab.get(k) not in have]
        missing_required = [k for k in missing_keys if k not in optional]
        explanation = graph_explanation(idx, ids)
        group = 'ready' if not missing_required else 'needs_shopping'
        max_imp = max(dish['ingredients'].values()) if dish['ingredients'] else 1.0
        
        result[group].append({
            'dish_id': dish['id'],
            'dish': dish['name'],
            'score': round(float(scores[idx]), 4),
            'score_percent': int(min(100, max(0, float(scores[idx]) * 100))),
            'matched': [names.get(k, k) for k in dish['ingredients'] if vocab.get(k) in have],
            'missing_required': [names.get(k, k) for k in missing_required],
            'missing_optional': [names.get(k, k) for k in missing_keys if k in optional],
            'missing_main_provisional': [names.get(k, k) for k in missing_required if dish['ingredients'][k] == max_imp],
            'input_utilization': round(float(coverage[0, idx]) * 100, 1),
            'category': explanation.get('category', 'Khác'),
            'kg_explanation': explanation,
            'url': dish.get('url'),
            'image_url': dish.get('image_url') or 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&w=400&q=80'
        })
        
    result['needs_shopping'].sort(key=lambda item: (
        len(item['missing_main_provisional']),
        len(item['missing_required']),
        -item['score']
    ))
    result['ready'] = result['ready'][:top_k]
    result['needs_shopping'] = result['needs_shopping'][:top_k]
    return result

@app.route('/')
def index():
    return render_template('index.html', categories=categories)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json() or {}
    ingredients = data.get('ingredients', [])
    top_k = int(data.get('top_k', 12))
    method = data.get('method', 'hybrid')
    category = data.get('category', 'all')
    assume_pantry = bool(data.get('assume_pantry', True))
    
    result = perform_recommend(
        ingredients=ingredients,
        top_k=top_k,
        method=method,
        category=category,
        assume_pantry=assume_pantry
    )
    return jsonify(result)

@app.route('/api/recipe/<dish_id>', methods=['GET'])
def api_recipe_detail(dish_id):
    dish = dish_details.get(dish_id)
    if not dish:
        return jsonify({'error': 'Không tìm thấy công thức món ăn'}), 404
        
    return jsonify({
        'id': dish.get('id'),
        'name': dish.get('name_vi'),
        'category': dish.get('category'),
        'servings': dish.get('servings', 4),
        'image_url': dish.get('image_url'),
        'url': dish.get('url'),
        'ingredients': [
            {
                'name': ing.get('name_vi', ''),
                'quantity': ing.get('quantity', ''),
                'unit': ing.get('unit', ''),
                'importance': ing.get('importance', 1)
            }
            for ing in dish.get('ingredients', [])
        ],
        'steps': dish.get('steps', [])
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({
        'num_recipes': len(recipes),
        'num_ingredients': len(vocab),
        'categories': categories,
        'best_coeff': list(best_coeff)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
