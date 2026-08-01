import os, cv2, regex, random, numpy as np, unicodedata
from PIL import Image, ImageDraw, ImageFont

# ── 1. Configuration (Path Logic for MyScriptsss) ──
# Points to .../Synthetic_Data/MyScriptsss
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Points to .../Synthetic_Data (Your Project Root)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Type_4_Stain_Fade_Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Verified Paths ───────────────────────────────
TEXT_SOURCES = [os.path.join(PROJECT_ROOT, "corpus", "GarudaPuranam", "garuda.txt")]
BASE_IMAGE   = os.path.join(PROJECT_ROOT, "backgrounds", "base_palm_leaf.png")
LOCAL_FONT   = os.path.join(PROJECT_ROOT, "GANs", "Nirmala.ttc")

MAPPING_BASE = os.path.join(PROJECT_ROOT, "mappings")
MAPPING_FILES = [
    os.path.join(MAPPING_BASE, "model_5_chars",    "all_progress_backup.txt"),
    os.path.join(MAPPING_BASE, "model_10_chars",   "all_progress_backup.txt"),
    os.path.join(MAPPING_BASE, "model_hand_chars", "all_progress_backup.txt"),
    os.path.join(MAPPING_BASE, "model_20_chars",   "all_progress_backup.txt"),
]

FIXED_HEIGHT, X_MARGIN, RIGHT_MARGIN, BASELINE_Y = 40, 17, 65, 35
CHAR_SPACING, WORD_SPACING = 5, 12

# ── 3. Helper Functions ────────────────────────────

def get_font(size):
    paths = [LOCAL_FONT, "C:/Windows/Fonts/Nirmala.ttc"]
    for fp in paths:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return None

def build_mapping():
    mapping = {}
    for map_file in MAPPING_FILES:
        if not os.path.exists(map_file): continue
        bd = os.path.dirname(map_file)
        with open(map_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    label = unicodedata.normalize('NFC', parts[1].strip())
                    path  = os.path.normpath(os.path.join(bd, parts[0].strip()))
                    if os.path.exists(path): mapping.setdefault(label, []).append(path)
    return mapping

def extract_real(img_path, h):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # Tiny blur to simulate ink weathering
    bin_img = cv2.GaussianBlur(bin_img, (3, 3), 0)
    ys, xs = np.where(bin_img > 0)
    if len(ys) == 0: return None
    crop = bin_img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = h / crop.shape[0]
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), h))
    return Image.fromarray(255 - resized)

def generate_synthetic(char, font, h):
    if not font: return None
    canv = Image.new("L", (h * 5, h * 3), 255); d = ImageDraw.Draw(canv)
    d.text((20, 20), char, font=font, fill=0); arr = np.array(canv)
    _, bin_img = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    arr = cv2.GaussianBlur(bin_img, (3, 3), 0)
    _, arr = cv2.threshold(arr, 110, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(arr > 0)
    if len(ys) == 0: return None
    crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = h / crop.shape[0]
    res = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), h))
    return Image.fromarray(255 - res)

def multiply_paste_faded(base_rgb, char_img, x, y, opacity):
    bw, bh = base_rgb.size; cw, ch = char_img.size
    x2, y2 = min(bw, x + cw), min(bh, y + ch)
    if x2 <= x or y2 <= y: return
    base_arr = np.array(base_rgb).astype(np.float32)
    char_arr = np.array(char_img.crop((0, 0, x2-x, y2-y))).astype(np.float32)
    
    alpha = (255 - char_arr) / 255.0 * opacity
    alpha = (alpha * np.random.uniform(0.60, 1.0, alpha.shape))[..., None]
    
    region = base_arr[y:y2, x:x2]
    base_arr[y:y2, x:x2] = np.clip(region * (1 - alpha), 0, 255)
    base_rgb.paste(Image.fromarray(base_arr.astype(np.uint8)), (0, 0))

def apply_solid_white_stains(canvas, text_limit_x):
    """Adds solid white stains on top of faded text."""
    draw = ImageDraw.Draw(canvas)
    for _ in range(random.randint(2, 3)): # Strips
        tx, cy = random.randint(X_MARGIN, text_limit_x), random.randint(15, 50)
        rx, ry = random.randint(40, 80), random.randint(3, 6)
        draw.ellipse([tx-rx, cy-ry, tx+rx, cy+ry], fill=(255, 255, 255))
    for _ in range(random.randint(15, 25)): # Surface specks
        cx, cy = random.randint(0, canvas.size[0]), random.randint(0, canvas.size[1])
        r = random.randint(1, 3)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 255))

# ── 4. Main Loop (1500 Faded + Stains) ─────────────

if __name__ == "__main__":
    mapping = build_mapping()
    syn_font = get_font(int(FIXED_HEIGHT * 1.5))
    if not os.path.exists(BASE_IMAGE):
        print(f"❌ Error: Background missing at {BASE_IMAGE}")
        exit()
    bg_template = cv2.imread(BASE_IMAGE)
    bg_rgb = cv2.cvtColor(bg_template, cv2.COLOR_BGR2RGB)
    
    if not os.path.exists(TEXT_SOURCES[0]):
        print(f"❌ Error: {TEXT_SOURCES[0]} not found.")
        exit()

    with open(TEXT_SOURCES[0], "r", encoding="utf-8") as f:
        all_words = f.read().split()

    word_idx, img_count = 0, 0
    while word_idx < len(all_words):
        canvas = Image.fromarray(bg_rgb.copy())
        curr_x, line_text = X_MARGIN, []

        while word_idx < len(all_words):
            word = all_words[word_idx]
            clusters = regex.findall(r'\X', word)
            word_imgs, word_w = [], 0
            
            # --- Sequence Shield ---
            word_successfully_built = True
            for c in clusters:
                c = unicodedata.normalize('NFC', c)
                img = extract_real(random.choice(mapping[c]), FIXED_HEIGHT) if c in mapping else None
                if img is None: 
                    img = generate_synthetic(c, syn_font, FIXED_HEIGHT)
                
                if img:
                    word_imgs.append(img)
                    word_w += img.size[0] + CHAR_SPACING
                else:
                    word_successfully_built = False
                    break

            if not word_successfully_built:
                word_idx += 1 # Skip broken word to maintain sequence
                continue

            # ROLL-OVER: Word stays at word_idx if it doesn't fit
            if curr_x + word_w > (bg_rgb.shape[1] - RIGHT_MARGIN):
                break 
            
            # Fading range for Mixed Logic
            word_opacity = random.uniform(0.15, 0.45)
            
            for img in word_imgs:
                multiply_paste_faded(canvas, img, curr_x, BASELINE_Y - (img.size[1]//2), word_opacity)
                curr_x += img.size[0] + CHAR_SPACING
            
            line_text.append(word)
            curr_x += WORD_SPACING
            word_idx += 1 # Increment word only after successful draw

        # Apply solid stains on TOP of faded ink
        apply_solid_white_stains(canvas, curr_x)

        name = f"faded_stain_{img_count:05d}"
        canvas.save(os.path.join(OUTPUT_DIR, f"{name}.png"))
        with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(line_text))
        
        img_count += 1
        if img_count >= 500: break # Set to your target for this script

    print(f"✅ Faded + Stain generation complete. Folder: {OUTPUT_DIR}")