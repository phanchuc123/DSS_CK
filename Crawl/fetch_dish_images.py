"""Enrich the recipe JSON with og:image URLs from the source recipe pages.

Run with a small --limit first. Respect the source website's robots.txt and
terms before collecting a larger set of images.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import sys
import time
from urllib.request import Request, urlopen
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent


class ImageMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.image_url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta" or self.image_url:
            return
        attributes = dict(attrs)
        key = attributes.get("property") or attributes.get("name")
        if key in {"og:image", "twitter:image"}:
            self.image_url = (attributes.get("content") or "").strip()


def find_image(url: str) -> str:
    request = Request(url, headers={"User-Agent": "FoodRecommendationPrototype/1.0"})
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")
    parser = ImageMetaParser()
    parser.feed(html)
    return parser.image_url


def download_image(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "FoodRecommendationPrototype/1.0"})
    with urlopen(request, timeout=30) as response:
        target.write_bytes(response.read())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dish_knowledge_base.json")
    parser.add_argument("--output", default="dish_knowledge_base_with_images.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--download", action="store_true", help="Save images under webapp/images")
    parser.add_argument(
        "--download-existing",
        action="store_true",
        help="Download existing HTTP image_url values instead of skipping them",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    dishes = json.loads(input_path.read_text(encoding="utf-8"))
    updated = 0

    for index, dish in enumerate(dishes):
        if index >= args.limit:
            break
        existing_image_url = str(dish.get("image_url") or "")
        if args.download_existing and existing_image_url.startswith(("http://", "https://")):
            image_url = existing_image_url
        elif dish.get("image_url") or not dish.get("url"):
            continue
        else:
            image_url = ""
        try:
            if not image_url:
                image_url = find_image(dish["url"])
            if image_url:
                if args.download:
                    images_dir = ROOT / "webapp" / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    suffix = Path(urlparse(image_url).path).suffix.lower()
                    suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
                    local_path = images_dir / f"{dish.get('id', index)}{suffix}"
                    download_image(image_url, local_path)
                    dish["image_url"] = f"images/{local_path.name}"
                else:
                    dish["image_url"] = image_url
                updated += 1
                print(f"[{index + 1}] {dish.get('name_vi', '')}: OK")
            else:
                print(f"[{index + 1}] {dish.get('name_vi', '')}: no og:image")
        except Exception as error:
            print(f"[{index + 1}] {dish.get('name_vi', '')}: {error}")
        time.sleep(0.5)

    output_path.write_text(
        json.dumps(dishes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã thêm {updated} ảnh vào {output_path}")


if __name__ == "__main__":
    main()
