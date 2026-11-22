import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import time
import os
import unicodedata
from html import unescape

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

def parse_dish_name_and_servings(h2_tag):
    """Lấy dish_name và servings từ h2 tag"""
    dish_name = h2_tag.find(string=True, recursive=False)
    if dish_name:
        dish_name = dish_name.strip()
        dish_name = re.sub(r'^Nguyên\s+liệu\s+làm\s+', '', dish_name, flags=re.IGNORECASE).strip()
    else:
        dish_name = ""
    
    servings = None
    small = h2_tag.find('small')
    if small:
        match = re.search(r'(\d+)', small.text)
        if match:
            servings = int(match.group(1))
    
    return dish_name, servings

def parse_quantity_unit(amount_text):
    """Tách quantity và unit từ text"""
    if not amount_text:
        return None, None
    
    amount_text = amount_text.strip()
    
    # "500 gram" -> 500, "gram"
    match = re.match(r'^(\d+(?:[.,]\d+)?)\s+(.+)$', amount_text)
    if match:
        return float(match.group(1).replace(',', '.')), match.group(2).strip()
    
    # "1/2 kg" -> 0.5, "kg"
    match = re.match(r'^(\d+)/(\d+)\s+(.+)$', amount_text)
    if match:
        return float(match.group(1)) / float(match.group(2)), match.group(3).strip()
    
    # "500" -> 500, None
    match = re.match(r'^(\d+(?:[.,]\d+)?)$', amount_text)
    if match:
        return float(match.group(1).replace(',', '.')), None
    
    return None, amount_text

def normalize_for_search(text):
    """Chuẩn hóa chữ để tìm các tiêu đề như 'Cách làm' trên trang."""
    text = unicodedata.normalize('NFKD', text or '')
    return ''.join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()

def clean_html_text(text):
    """Giải mã HTML entity bị mã hóa lặp và loại bỏ HTML còn sót lại."""
    decoded = text or ''
    for _ in range(3):
        next_value = unescape(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    decoded = BeautifulSoup(decoded, 'html.parser').get_text('\n')
    decoded = re.sub(r'\n[ \t]+', '\n', decoded)
    decoded = re.sub(r'\n{3,}', '\n\n', decoded)
    return decoded.strip()

def parse_cooking_steps(soup):
    """Lấy bước nấu từ JSON-LD hoặc phần tiêu đề 'Cách làm' trên bài viết."""
    # Nhiều trang công thức nhúng hướng dẫn trong JSON-LD chuẩn schema.org.
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            payload = json.loads(script.string or script.get_text())
            objects = payload if isinstance(payload, list) else [payload]
            for obj in objects:
                instructions = obj.get('recipeInstructions') if isinstance(obj, dict) else None
                if not instructions:
                    continue
                if isinstance(instructions, str):
                    return [clean_html_text(instructions)]
                steps = []
                for item in instructions:
                    text = item.get('text', '') if isinstance(item, dict) else str(item)
                    if text and text.strip():
                        steps.append(clean_html_text(text))
                if steps:
                    return steps
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    # Fallback: đọc các đoạn p/li sau tiêu đề Cách làm/Thực hiện/Hướng dẫn.
    headings = soup.find_all(['h2', 'h3', 'h4'])
    keywords = ('cach lam', 'thuc hien', 'huong dan nau', 'che bien')
    for heading in headings:
        title = normalize_for_search(heading.get_text(' ', strip=True))
        if not any(keyword in title for keyword in keywords):
            continue
        steps = []
        for node in heading.find_all_next(['h2', 'h3', 'h4', 'p', 'li']):
            if node.name in ('h2', 'h3', 'h4'):
                break
            text = node.get_text(' ', strip=True)
            if text and text not in steps:
                steps.append(clean_html_text(text))
        if steps:
            return steps
    return []

def crawl_recipe(url):
    """Crawl 1 recipe"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        staple_div = soup.find('div', class_='staple')
        if not staple_div:
            return None
        
        h2 = staple_div.find('h2')
        dish_name, servings = parse_dish_name_and_servings(h2) if h2 else ("", None)
        steps = parse_cooking_steps(soup)
        
        ingredients = []
        for span in staple_div.find_all('span'):
            # Lấy tên nguyên liệu
            ingredient_name = ' '.join([t.strip() for t in span.find_all(string=True, recursive=False)]).strip()
            
            # Lấy quantity từ small
            small = span.find('small')
            amount_text = small.text.strip() if small else ""
            quantity, unit = parse_quantity_unit(amount_text)
            
            # Kiểm tra nếu là "Gia vị thông dụng"
            em_tag = span.find('em')
            if em_tag and 'gia vị' in ingredient_name.lower():
                # Tách các gia vị từ thẻ em
                spices_text = re.sub(r'^\(|\)$', '', em_tag.text.strip())
                for spice in re.split(r'[,/]', spices_text):
                    spice = spice.strip().capitalize()
                    if spice:
                        ingredients.append({'name': spice, 'quantity': quantity, 'unit': unit})
            elif ingredient_name:
                ingredients.append({'name': ingredient_name, 'quantity': quantity, 'unit': unit})
        
        return {
            'dish_name': dish_name,
            'url': url,
            'servings': servings,
            'ingredients': ingredients,
            'steps': steps
        }
    
    except Exception as e:
        print(f"  Lỗi: {e}")
        return None

def main():
    input_path = 'data/recipe_urls.csv'
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Không tìm thấy {input_path}. Hãy chạy 1-crawl_dish_urls.py trước."
        )
    if os.path.getsize(input_path) == 0:
        raise ValueError(
            f"{input_path} đang rỗng. Kiểm tra Chrome/Edge WebDriver rồi chạy lại 1-crawl_dish_urls.py."
        )

    df = pd.read_csv(input_path)
    if df.empty or 'url' not in df.columns:
        raise ValueError(
            f"{input_path} không có URL hợp lệ. Hãy chạy lại 1-crawl_dish_urls.py."
        )

    os.makedirs('data', exist_ok=True)
    
    recipes = []
    for i in range(len(df)):
        row = df.iloc[i]
        url = row['url']
        
        print(f"\n[{i+1}/{len(df)}] {url.split('/')[-1][:60]}")
        
        recipe = crawl_recipe(url)
        if recipe:
            recipe['category'] = row['category']
            recipes.append(recipe)
        
        time.sleep(0.5)
    
    with open('data/recipes_detail.json', 'w', encoding='utf-8') as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✓ Đã crawl {len(recipes)} món")
    print("=" * 60)

if __name__ == "__main__":
    main()
