import os, subprocess, unicodedata

# ── 1. Configuration ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_FOLDER = os.path.join(BASE_DIR, "Final_data")
os.makedirs(FINAL_FOLDER, exist_ok=True)

# THE FOLDER YOU CREATED
SCRIPTS_DIR = os.path.join(BASE_DIR, "the_Finale")
CORPUS_PATH = os.path.join(BASE_DIR, "corpus", "Mahabharatham", "gita.txt")

# FIXED NAMES: Matching your 'the_Finale' folder exactly
SCRIPTS = [
    "type_1.py", "type_2.py", "type_3.py", "type_4.py",
    "type_5.py", "type_5d.py", "type_5dd.py",
    "type_6.py", "type_6d.py", "type_6dd.py",
    "type_7.py"
]

# ── 2. Load & Slice Corpus ──────────────────────────
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    # Ensuring 100% Sequence Integrity (NFC Normalization)
    all_words = unicodedata.normalize('NFC', f.read()).split()

# Each script gets an equal slice of the 12,000 lines
words_per_batch = len(all_words) // len(SCRIPTS)

# ── 3. Sequential Execution ─────────────────────
global_idx = 0
word_pointer = 0

print(f"🚀 Starting the Finale Production for {len(all_words)} words...")

for script_name in SCRIPTS:
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    # Safety Check: Does the file actually exist?
    if not os.path.exists(script_path):
        print(f"❌ ERROR: {script_name} not found in the_Finale! Check spelling.")
        continue

    # Calculate word slice boundaries
    end_pointer = word_pointer + words_per_batch
    if script_name == SCRIPTS[-1]: # Final script takes all remaining words
        end_pointer = len(all_words)
    
    current_slice = all_words[word_pointer:end_pointer]
    
    # Save temporary slice for the script to read
    temp_slice = os.path.join(BASE_DIR, "current_slice.txt")
    with open(temp_slice, "w", encoding="utf-8") as f:
        f.write(" ".join(current_slice))

    print(f"🎬 Running {script_name} | Words: {word_pointer} to {end_pointer} | Start ID: {global_idx:05d}")

    # RUN THE SCRIPT
    # Arguments: [Temp Text Path] [Current Global ID] [Output Directory]
    subprocess.run(["python", script_path, temp_slice, str(global_idx), FINAL_FOLDER])

    # UPDATE COUNTERS
    # We count how many PNGs are in the folder now to set the next start ID
    global_idx = len([f for f in os.listdir(FINAL_FOLDER) if f.endswith('.png')])
    word_pointer = end_pointer

print(f"\n✅ SUCCESS,! Your 'Final_data' is ready with all 11 types.")