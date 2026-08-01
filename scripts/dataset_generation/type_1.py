import os, cv2, regex, random, numpy as np, unicodedata
from PIL import Image, ImageDraw, ImageFont

# ── 1. Type 1 Switch ───────────────────────────────
USE_GAPS = True 

# ── 2. Configuration ────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "Type_1_Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to your Garuda folder inside the project root
TEXT_SOURCES  = [os.path.join(PROJECT_ROOT, "corpus", "Mahabharatham", "gita.txt")]
BASE_IMAGE    = os.path.join(PROJECT_ROOT, "backgrounds", "base_palm_leaf.png")
LOCAL_FONT    = os.path.join(PROJECT_ROOT, "GANs", "Nirmala.ttc")

MAPPING_FILES = [
    os.path.join(PROJECT_ROOT, "mappings", "model_5_chars",    "all_progress_backup.txt"),
    os.path.join(PROJECT_ROOT, "mappings", "model_10_chars",   "all_progress_backup.txt"),
    os.path.join(PROJECT_ROOT, "mappings", "model_hand_chars", "all_progress_backup.txt"),
    os.path.join(PROJECT_ROOT, "mappings", "model_20_chars",   "all_progress_backup.txt"),
]

# ── 3. Spacing Settings ─────────────────────────────
FIXED_HEIGHT, X_MARGIN, RIGHT_MARGIN, BASELINE_Y = 40, 17, 65, 35
CHAR_SPACING, WORD_SPACING = 5, 12

# Physical Gap Logic (Logic C)
GAP_COORDS    = [1927, 3709] if USE_GAPS else []
BIG_GAP_WIDTH = 160 

# ── 4. Helper Functions (Exactly as you provided) ──

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
    ys, xs = np.where(bin_img > 0)
    if len(ys) == 0: return None
    crop = bin_img[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = h / crop.shape[0]
    resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), h))
    return Image.fromarray(255 - resized)

def multiply_paste_rgb(base_rgb, char_img, x, y):
    bw, bh = base_rgb.size; cw, ch = char_img.size
    x2, y2 = min(bw, x + cw), min(bh, y + ch)
    if x2 <= x or y2 <= y: return
    base_arr = np.array(base_rgb).astype(np.float32)
    char_arr = np.array(char_img.crop((0, 0, x2-x, y2-y))).astype(np.float32)
    alpha = (255 - char_arr) / 255.0 * 0.45
    alpha = (alpha * np.random.uniform(0.65, 1.0, alpha.shape))[..., None]
    region = base_arr[y:y2, x:x2]
    base_arr[y:y2, x:x2] = np.clip(region * (1 - alpha), 0, 255)
    base_rgb.paste(Image.fromarray(base_arr.astype(np.uint8)), (0, 0))

# ── 5. Main Loop (Bucket 1: Gaps) ──────────────────

if __name__ == "__main__":
    mapping = build_mapping()
    syn_font = ImageFont.truetype(LOCAL_FONT, int(FIXED_HEIGHT * 1.5))
    bg_template = cv2.imread(BASE_IMAGE)
    bg_rgb = cv2.cvtColor(bg_template, cv2.COLOR_BGR2RGB)
    
    with open(TEXT_SOURCES[0], "r", encoding="utf-8") as f:
        all_words = f.read().split()

    word_idx, img_count = 0, 0
    while word_idx < len(all_words):
        canvas = Image.fromarray(bg_rgb.copy())
        curr_x, line_text = X_MARGIN, []

        while word_idx < len(all_words):
            word = all_words[word_idx]
            clusters = regex.findall(r'\X', word)
            w_imgs, w_width = [], 0
            for c in clusters:
                c = unicodedata.normalize('NFC', c)
                img = extract_real(random.choice(mapping[c]), FIXED_HEIGHT) if c in mapping else None
                if not img: # Synthetic fallback logic as provided
                    canv = Image.new("L", (FIXED_HEIGHT*5, FIXED_HEIGHT*3), 255)
                    d = ImageDraw.Draw(canv)
                    d.text((20, 20), c, font=syn_font, fill=0)
                    arr = np.array(canv)
                    _, bin_img = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    ys, xs = np.where(bin_img > 0)
                    if len(ys) > 0:
                        crop = bin_img[ys.min():ys.max()+1, xs.min():xs.max()+1]
                        scale = FIXED_HEIGHT / crop.shape[0]
                        res = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), FIXED_HEIGHT))
                        img = Image.fromarray(255 - res)
                if img:
                    w_imgs.append(img); w_width += img.size[0] + CHAR_SPACING

            # --- GAP LOGIC ---
            tx = curr_x
            for b in GAP_COORDS:
                if tx < b and (tx + w_width) > b:
                    tx += BIG_GAP_WIDTH 

            if tx + w_width > (bg_rgb.shape[1] - RIGHT_MARGIN):
                break 
            
            curr_x = tx 
            for img in w_imgs:
                multiply_paste_rgb(canvas, img, curr_x, BASELINE_Y - (img.size[1]//2))
                curr_x += img.size[0] + CHAR_SPACING
            
            line_text.append(word); curr_x += WORD_SPACING; word_idx += 1

        suffix = "gaps" if USE_GAPS else "normal"
        name = f"type1_{suffix}_{img_count:05d}"
        canvas.save(os.path.join(OUTPUT_DIR, f"{name}.png"))
        with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f: f.write(" ".join(line_text))
        
        img_count += 1
        if img_count >= 2000: break # Goal for Bucket 1 Gaps