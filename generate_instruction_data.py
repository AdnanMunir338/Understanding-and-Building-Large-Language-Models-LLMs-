import os
import re
import json
import random
import requests

os.environ["HF_HOME"]            = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mimer/NOBACKUP/groups/naiss2024-22-1298/Adnan/hf_cache"

# ═══════════════════════════════════════════════════════════════════════════════
#  Download Gutenberg books
# ═══════════════════════════════════════════════════════════════════════════════

BOOKS = {
    "war_and_peace":      "https://www.gutenberg.org/files/2600/2600-0.txt",
    "pride_prejudice":    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "moby_dick":          "https://www.gutenberg.org/files/2701/2701-0.txt",
    "tale_two_cities":    "https://www.gutenberg.org/files/98/98-0.txt",
    "sherlock_holmes":    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "dracula":            "https://www.gutenberg.org/files/345/345-0.txt",
    "frankenstein":       "https://www.gutenberg.org/files/84/84-0.txt",
    "great_expectations": "https://www.gutenberg.org/files/1400/1400-0.txt",
}

BOOK_AUTHORS = {
    "war_and_peace":      "Leo Tolstoy",
    "pride_prejudice":    "Jane Austen",
    "moby_dick":          "Herman Melville",
    "tale_two_cities":    "Charles Dickens",
    "sherlock_holmes":    "Arthur Conan Doyle",
    "dracula":            "Bram Stoker",
    "frankenstein":       "Mary Shelley",
    "great_expectations": "Charles Dickens",
}

BOOK_TITLES = {
    "war_and_peace":      "War and Peace",
    "pride_prejudice":    "Pride and Prejudice",
    "moby_dick":          "Moby Dick",
    "tale_two_cities":    "A Tale of Two Cities",
    "sherlock_holmes":    "The Adventures of Sherlock Holmes",
    "dracula":            "Dracula",
    "frankenstein":       "Frankenstein",
    "great_expectations": "Great Expectations",
}


def load_books(cache_dir="./data"):
    os.makedirs(cache_dir, exist_ok=True)
    books_text = {}
    for name, url in BOOKS.items():
        fpath = os.path.join(cache_dir, f"{name}.txt")
        if not os.path.exists(fpath):
            print(f"  Downloading {name}...")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(r.text)
                books_text[name] = r.text
            except Exception as e:
                print(f"  Skipping {name}: {e}")
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                books_text[name] = f.read()
    return books_text


def clean_gutenberg(text):
    start_match = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    end_match = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    text = re.sub(r"^[A-Z][A-Z\s\.\-]{5,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  Extract clean paragraphs from a book
# ═══════════════════════════════════════════════════════════════════════════════

def extract_paragraphs(text, min_words=40, max_words=120):
    """Extract paragraphs of a reasonable length for instruction generation."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result = []
    for p in paragraphs:
        word_count = len(p.split())
        if min_words <= word_count <= max_words:
            # clean up internal newlines
            p = re.sub(r"\s+", " ", p)
            result.append(p)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Generate instruction-response pairs from book paragraphs
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pairs_from_book(book_name, text, n_pairs=500):
    """
    Auto-generate instruction-response pairs from a single book.
    Uses 6 instruction templates per paragraph sampled randomly.
    """
    title  = BOOK_TITLES[book_name]
    author = BOOK_AUTHORS[book_name]
    paragraphs = extract_paragraphs(text)

    if not paragraphs:
        return []

    # Instruction templates — each takes the paragraph as the response
    # or uses a split of the paragraph as prompt→continuation
    def make_pairs(para):
        words     = para.split()
        half      = " ".join(words[:len(words)//2])   # first half as prompt
        pairs = []

        # 1. Continue the passage
        pairs.append({
            "instruction": f"Continue the following passage in the style of {author}:\n\n{half}",
            "response":    " ".join(words[len(words)//2:])
        })

        # 2. Write in the style of the author
        pairs.append({
            "instruction": f"Write a short passage in the literary style of {author}, "
                           f"the author of {title}.",
            "response":    para
        })

        # 3. Summarise the passage
        summary_words = words[:15]
        pairs.append({
            "instruction": f"Summarise the following passage from {title} by {author} "
                           f"in one sentence:\n\n{para}",
            "response":    "This passage " + " ".join(summary_words) + "..."
        })

        # 4. Describe the mood
        pairs.append({
            "instruction": f"What is the mood or tone of the following passage from {title}?\n\n{para}",
            "response":    f"The passage conveys a {random.choice(['melancholic', 'tense', 'romantic', 'dramatic', 'reflective', 'mysterious'])} "
                           f"tone, characteristic of {author}'s writing style in {title}."
        })

        # 5. Rewrite more formally
        pairs.append({
            "instruction": f"Rewrite the following passage in a formal 19th century literary style:\n\n{half}",
            "response":    para
        })

        # 6. Answer a question about the text
        pairs.append({
            "instruction": f"Who wrote {title} and what is it about?",
            "response":    f"{title} was written by {author}. It is a classic 19th century novel "
                           f"known for its rich characterisation and narrative depth."
        })

        return pairs

    # Sample paragraphs and generate pairs
    random.shuffle(paragraphs)
    all_pairs = []
    for para in paragraphs:
        if len(all_pairs) >= n_pairs:
            break
        all_pairs.extend(make_pairs(para))

    return all_pairs[:n_pairs]


# ═══════════════════════════════════════════════════════════════════════════════
#  Download Alpaca dataset (general instruction-following)
# ═══════════════════════════════════════════════════════════════════════════════

def load_alpaca(n_samples=5000):
    """
    Download Stanford Alpaca dataset — 52K general instruction-response pairs.
    We use a subset (5000) to keep training time manageable.
    """
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    cache_path = "./data/alpaca_data.json"

    if not os.path.exists(cache_path):
        print("  Downloading Alpaca dataset...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(cache_path, "w") as f:
            f.write(r.text)

    with open(cache_path, "r") as f:
        data = json.load(f)

    # Alpaca format: {"instruction": ..., "input": ..., "output": ...}
    # Convert to our format: {"instruction": ..., "response": ...}
    pairs = []
    for item in data:
        instruction = item["instruction"]
        if item.get("input", "").strip():
            instruction = instruction + "\n\n" + item["input"]
        pairs.append({
            "instruction": instruction,
            "response":    item["output"]
        })

    random.shuffle(pairs)
    print(f"  Loaded {len(pairs):,} Alpaca pairs → using {n_samples:,}")
    return pairs[:n_samples]


# ═══════════════════════════════════════════════════════════════════════════════
#  Format for GPT-2 instruction fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

def format_pair(instruction, response):
    """
    Format a single instruction-response pair as a training string.
    GPT-2 has no special tokens, so we use a clear text template.
    The model learns to generate everything after '### Response:'
    """
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}<|endoftext|>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(42)
    os.makedirs("./data", exist_ok=True)

    all_pairs = []

    # ── 1. Literary pairs from 8 books ───────────────────────────────────────
    print("\nGenerating literary instruction pairs from books...")
    books_text = load_books()
    for name, text in books_text.items():
        clean = clean_gutenberg(text)
        pairs = generate_pairs_from_book(name, clean, n_pairs=400)
        all_pairs.extend(pairs)
        print(f"  {BOOK_TITLES[name]}: {len(pairs)} pairs")

    print(f"Total literary pairs: {len(all_pairs):,}")

    # ── 2. Alpaca general instruction pairs ───────────────────────────────────
    print("\nLoading Alpaca general instruction pairs...")
    alpaca_pairs = load_alpaca(n_samples=5000)
    all_pairs.extend(alpaca_pairs)

    # ── 3. Shuffle combined dataset ───────────────────────────────────────────
    random.shuffle(all_pairs)
    print(f"\nTotal combined pairs: {len(all_pairs):,}")

    # ── 4. Format and save ────────────────────────────────────────────────────
    formatted = [
        {
            "text": format_pair(p["instruction"], p["response"]),
            "instruction": p["instruction"],
            "response": p["response"],
        }
        for p in all_pairs
    ]

    out_path = "./data/instruction_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(formatted):,} instruction pairs → {out_path}")
    print("\nSample entry:")
    print(json.dumps(formatted[0], indent=2)[:500])