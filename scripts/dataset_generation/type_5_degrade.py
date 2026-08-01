import os, cv2, regex, random, numpy as np, unicodedata
from PIL import Image, ImageDraw, ImageFont

# ── 1. Configuration (Path Logic for MyScriptsss) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Type_5_Degraded_Pure_Output")
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
DEGRADE_SIGMA = 48.0 # Extreme texture noise

# ── 3. Helper Functions ────────────────────────────

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

def add_salt_and_pepper(pil_img, prob=0.04):
    """Adds dirt specks to simulate physical deterioration."""
    arr = np.array(pil_img)
    noise = np.random.rand(*arr.shape[:2])
    arr[noise < (prob/2)] = 0     # Pepper (Dirt)
    arr[noise > (1 - prob/2)] = 255 # Salt (Fiber loss)
    return Image.fromarray(arr)

def add_extreme_grains(pil_img):
    """Adds high-frequency Gaussian noise for aged texture."""
    arr = np.array(pil_img).astype(np.float32)
    raw_noise = np.random.normal(0, DEGRADE_SIGMA, arr.shape[:2])
    mottled_noise = cv2.GaussianBlur(raw_noise, (15, 15), 0)
    arr += mottled_noise[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def multiply_paste_rgb(base_rgb, char_img, x, y):
    bw, bh = base_rgb.size; cw, ch = char_img.size
    x2, y2 = min(bw, x + cw), min(bh, y + ch)
    if x2 <= x or y2 <= y: return
    base_arr = np.array(base_rgb).astype(np.float32)
    char_arr = np.array(char_img.crop((0, 0, x2-x, y2-y))).astype(np.float32)
    alpha = (255 - char_arr) / 255.0 * 0.48
    alpha = (alpha * np.random.uniform(0.65, 1.0, alpha.shape))[..., None]
    region = base_arr[y:y2, x:x2]
    base_arr[y:y2, x:x2] = np.clip(region * (1 - alpha), 0, 255)
    base_rgb.paste(Image.fromarray(base_arr.astype(np.uint8)), (0, 0))

# ── 4. Main Loop ──────────────────────────────────

if __name__ == "__main__":
    mapping = build_mapping()
    syn_font = ImageFont.truetype(LOCAL_FONT, int(FIXED_HEIGHT * 1.5))
    bg_template = cv2.imread(BASE_IMAGE)
    bg_rgb = cv2.cvtColor(bg_template, cv2.COLOR_BGR2RGB)
    
    with open(TEXT_SOURCES[0], "r", encoding="utf-8") as f:
        all_words = f.read().split()

    word_idx, img_count = 0, 0
    while word_idx < len(all_words):
        # 1. Apply degradation to background first
        canvas = add_extreme_grains(Image.fromarray(bg_rgb.copy()))
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
                if img:
                    word_imgs.append(img); word_w += img.size[0] + CHAR_SPACING
                else:
                    word_successfully_built = False
                    break

            if not word_successfully_built:
                word_idx += 1
                continue

            if curr_x + word_w > (bg_rgb.shape[1] - RIGHT_MARGIN): break 
            
            for img in word_imgs:
                multiply_paste_rgb(canvas, img, curr_x, BASELINE_Y-(img.size[1]//2))
                curr_x += img.size[0] + CHAR_SPACING
            
            line_text.append(word); curr_x += WORD_SPACING; word_idx += 1

        # 2. Final Salt and Pepper layer over everything
        canvas = add_salt_and_pepper(canvas)
        
        name = f"degraded_pure_{img_count:05d}"
        canvas.save(os.path.join(OUTPUT_DIR, f"{name}.png"))
        with open(os.path.join(OUTPUT_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(line_text))
        
        img_count += 1
        if img_count >= 500: break