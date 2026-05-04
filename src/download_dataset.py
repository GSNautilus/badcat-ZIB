"""Download a small curated dataset of public-domain artworks from Wikimedia Commons.

Each entry has a clear caption. Variety chosen to give the adapter
meaningful style abstraction signal:
- Renaissance painting
- Impressionism, Post-impressionism
- Japanese woodblock prints
- Ancient/medieval
- Pre-1929 photography
- Sculpture photographs

Held-out for testing (NOT in this list): Van Gogh's Starry Night (already saved).
"""
from __future__ import annotations

import os
import json
import urllib.parse
import urllib.request


OUT_IMG_DIR = "data/wikimedia_train"
OUT_META = "data/wikimedia_train/captions.json"
UA = "ZImageStyleAdapter/0.1 (azrack@gmail.com) urllib"

# (filename_on_wikimedia, caption_for_training)
DATASET = [
    # --- Famous paintings, single subject ---
    ("Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg",
     "the Mona Lisa portrait painting by Leonardo da Vinci"),
    ("Vincent van Gogh - Sunflowers - VGM F458.jpg",
     "an oil painting of sunflowers in a vase by Van Gogh"),
    ("Vincent van Gogh - Wheatfield with crows - Google Art Project.jpg",
     "a Van Gogh oil painting of a wheatfield with crows"),
    ("Vincent van Gogh - Cafe Terrace at Night (Yorck).jpg",
     "Van Gogh's Cafe Terrace at Night painting"),
    ("Edvard Munch, 1893, The Scream, oil, tempera and pastel on cardboard, 91 x 73 cm, National Gallery of Norway.jpg",
     "Edvard Munch's painting The Scream"),
    ("Whistlers Mother high res.jpg",
     "the painting Whistler's Mother"),
    ("Grant Wood - American Gothic - Google Art Project.jpg",
     "the painting American Gothic by Grant Wood"),
    ("Sandro Botticelli - La nascita di Venere - Google Art Project - edited.jpg",
     "Botticelli's painting The Birth of Venus"),
    ("Johannes Vermeer (1632-1675) - The Girl With The Pearl Earring (1665).jpg",
     "Vermeer's painting Girl with a Pearl Earring"),
    ("Tsunami by hokusai 19th century.jpg",
     "the Japanese woodblock print The Great Wave off Kanagawa by Hokusai"),

    # --- Impressionism ---
    ("Claude Monet, Impression, soleil levant.jpg",
     "Monet's impressionist painting Impression Sunrise"),
    ("Pierre-Auguste Renoir, Le Moulin de la Galette.jpg",
     "Renoir's impressionist painting Bal du moulin de la Galette"),
    ("Edgar Degas - The Ballet Class - Google Art Project.jpg",
     "Degas's painting of a ballet class"),
    ("Claude Monet - Water Lilies - 1906, Ryerson.jpg",
     "Monet's impressionist painting of water lilies"),

    # --- More post-impressionism, 19c painting ---
    ("Paul Cezanne, Still Life with a Curtain.jpg",
     "Paul Cezanne's still life with curtain and fruit"),
    ("Paul Gauguin 137.jpg",
     "a Gauguin painting of Tahitian women"),
    ("Eugène Delacroix - Le 28 Juillet. La Liberté guidant le peuple.jpg",
     "Delacroix's painting Liberty Leading the People"),
    ("Caspar David Friedrich - Wanderer above the sea of fog.jpg",
     "Caspar David Friedrich's romantic painting Wanderer Above the Sea of Fog"),

    # --- Japanese woodblock variety ---
    ("Hokusai-fuji-koryo-bessho.jpg",
     "a Hokusai woodblock print of Mount Fuji"),
    ("Hiroshige - Plum Park in Kameido.jpg",
     "a Hiroshige woodblock print of a plum park in Kameido"),

    # --- Renaissance / earlier ---
    ("Da Vinci Vitruve Luc Viatour.jpg",
     "Leonardo da Vinci's drawing the Vitruvian Man"),
    ("Michelangelo - Creation of Adam (cropped).jpg",
     "Michelangelo's painting The Creation of Adam"),
    ("Raffael 040.jpg",
     "Raphael's painting The School of Athens"),
    ("Pieter Bruegel the Elder - The Tower of Babel (Vienna) - Google Art Project - edited.jpg",
     "Bruegel's painting The Tower of Babel"),

    # --- Pre-1929 photography (B&W) ---
    ("Eugene atget paris ca 1900.jpg",
     "an early black and white photograph of Paris by Eugene Atget"),
    ("Migrant Mother, Nipomo, California, 1936 - Dorothea Lange.jpg",
     "Dorothea Lange's photograph Migrant Mother"),

    # --- Symbolism / Art Nouveau ---
    ("Gustav Klimt 016.jpg",
     "Gustav Klimt's painting The Kiss in golden art nouveau style"),
    ("Alfons Mucha - 1896 - Salon des Cent.jpg",
     "an Alphonse Mucha art nouveau poster"),

    # --- Misc styles ---
    ("Pieter Bruegel d. Ä. 037.jpg",
     "Bruegel's painting Hunters in the Snow"),
    ("Henri Rousseau - Tiger in a Tropical Storm.jpg",
     "Henri Rousseau's painting Tiger in a Tropical Storm"),
    ("Caravaggio - The Calling of Saint Matthew.jpg",
     "Caravaggio's painting The Calling of Saint Matthew"),
]


def safe_filename(filename: str) -> str:
    """Sanitize a Wikimedia filename to a local filename."""
    base = filename.split(".")[0].replace(" ", "_").replace(",", "")[:60]
    ext = "." + filename.split(".")[-1].lower()
    return base + ext


def download_one(filename: str, dest: str, width: int = 768) -> bool:
    """Download via Special:FilePath URL pattern."""
    url = (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        + urllib.parse.quote(filename)
        + f"?width={width}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if len(data) < 5000:
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"  ERR {filename}: {e}")
        return False


def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    captions = {}
    ok, fail = 0, 0
    for i, (fn, cap) in enumerate(DATASET):
        dest_name = f"{i:02d}_{safe_filename(fn)}"
        dest = os.path.join(OUT_IMG_DIR, dest_name)
        if os.path.exists(dest) and os.path.getsize(dest) > 5000:
            print(f"[{i:02d}] skip (have): {dest_name}")
            captions[dest_name] = cap
            ok += 1
            continue
        success = download_one(fn, dest, width=768)
        if success:
            print(f"[{i:02d}] ok:   {dest_name}")
            captions[dest_name] = cap
            ok += 1
        else:
            print(f"[{i:02d}] FAIL: {fn}")
            fail += 1
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    print(f"\nTotal: {ok} ok, {fail} failed.")
    print(f"Captions written to {OUT_META}")


if __name__ == "__main__":
    main()
