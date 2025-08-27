import os, json, re, random
from datetime import datetime
from typing import List
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

SYSTEM_INSTRUCTIONS = (
    "You generate fictional people. "
    "Output ONLY a JSON array with no prose, no code fences."
)

USER_TEMPLATE = (
    "Generate {count} people as a JSON array. Each item must be an object with keys:\n"
    '  "firstName", "lastName", "dateOfBirth", "zipCode", "notes"\n'
    "Rules:\n"
    "- dateOfBirth: between 1920-01-01 and 2024-12-31, format YYYY-MM-DD.\n"
    "- zipCode: exactly 5 digits (allow leading zeros only if realistic US ZIPs).\n"
    "- notes: exactly one sentence (20–60 chars) ending with a period; detail such as spouse, profession, interest, etc..\n"
    "Return only the JSON array, nothing else."
)

def clamp_dob(d: str) -> str:
    """Clamp to 1920-01-01 .. 2024-12-31 and coerce invalid days to a safe range."""
    try:
        y, m, day = map(int, d.split("-"))
    except Exception:
        # random safe date if badly formatted
        y = random.randint(1920, 2024)
        m = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{y:04d}-{m:02d}-{day:02d}"
    y = min(max(y, 1920), 2024)
    m = min(max(m, 1), 12)
    day = min(max(day, 1), 28)
    return f"{y:04d}-{m:02d}-{day:02d}"

def sanitize_zip(z: str) -> str:
    z = re.sub(r"\D", "", z or "")
    if len(z) != 5:
        z = f"{random.randint(10000, 99999)}"
    return z

def one_sentence(note: str) -> str:
    note = (note or "").strip()
    # Remove newlines, enforce trailing single period, trim length.
    note = re.sub(r"\s+", " ", note)
    if len(note) < 20:
        note = "Nothing notable."
    if not note.endswith("."):
        note += "."
    while note.endswith(".."):
        note = note[:-1]
    return note[:60]

def parse_json_strict(text: str):
    # find first JSON array in text (in case the model added stray chars)
    m = re.search(r"\[\s*{", text, re.S)
    if not m:
        raise ValueError("No JSON array found in model output.")
    start = m.start()
    # naive bracket balance to slice array
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                arr = text[start : i + 1]
                return json.loads(arr)
    raise ValueError("Unbalanced JSON array in model output.")

def call_model(count: int) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = USER_TEMPLATE.format(count=count)

    # Try Responses API first
    try:
        resp = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=prompt,
        )
        return resp.output_text
    except Exception:
        assert(False)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content

def generate_people(count: int = 10, outfile: str = "people.json"):
    raw = call_model(count)
    data = parse_json_strict(raw)

    # Load existing array (or start fresh if file missing/corrupt)
    try:
        with open(outfile, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    # Validate & repair
    cleaned = []
    for p in data:
        first = (p.get("firstName") or "").strip().title()[:12]
        last  = (p.get("lastName") or "").strip().title()[:12]
        dob   = clamp_dob(p.get("dateOfBirth") or "")
        zipc  = sanitize_zip(p.get("zipCode") or "")
        notes = one_sentence(p.get("notes") or "")
        key = (first, last, dob, zipc)
        cleaned.append({
            "firstName": first or "Alex",
            "lastName": last or "Rivera",
            "dateOfBirth": dob,
            "zipCode": zipc,
            "notes": notes,
        })

    cleaned.extend(existing)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(cleaned)} people to {outfile}")

if __name__ == "__main__":
    while True:
        generate_people(count=10, outfile="people.json")
