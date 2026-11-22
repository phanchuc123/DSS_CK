import json
from unidecode import unidecode


# Phan loai don gian, khong can mo hinh Qwen.
CATEGORY_KEYWORDS = {
    "seasonings": ["muoi", "duong", "nuoc mam", "tieu", "ot", "toi", "dau an", "hat nem"],
    "fresh_meat": ["thit", "ga", "bo", "heo", "lon", "vit", "suon", "ba chi"],
    "seafood_&_fish_balls": ["tom", "ca ", "muc", "ngheu", "cua", "ghe", "so diep"],
    "vegetables": ["rau", "ca chua", "carot", "hanh", "gung", "nam", "bi", "khoai", "bap", "ngo"],
    "fresh_fruits": ["chuoi", "tao", "cam", "xoai", "dua", "chanh", "buoi", "nho"],
    "grains_staples": ["gao", "bot mi", "mi", "bun", "pho", "nui", "mien"],
    "milk": ["sua", "pho mai", "bo"],
}


def normalize_text(text):
    return unidecode(text).lower().strip()


def classify_category(ingredient):
    normalized = normalize_text(ingredient)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "others"


def build_kb():
    with open("data/unique_ingredients.json", "r", encoding="utf-8") as f:
        ingredients = json.load(f)

    kb = []
    for idx, ingredient in enumerate(ingredients, 1):
        kb.append({
            "id": f"ingre{idx:05d}",
            "name_vi": ingredient,
            "name_normalized": normalize_text(ingredient),
            "category": classify_category(ingredient),
            "type": "ingredient",
        })

    with open("ingredient_knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"Completed: {len(kb)} ingredients")
    print("Saved to: ingredient_knowledge_base.json")


if __name__ == "__main__":
    build_kb()
