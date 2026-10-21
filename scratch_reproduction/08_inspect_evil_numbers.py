import json
from pathlib import Path
from urllib.request import urlretrieve


URL = (
    "https://raw.githubusercontent.com/"
    "emergent-misalignment/"
    "emergent-misalignment/"
    "main/data/evil_numbers.jsonl"
)

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "evil_numbers.jsonl"


# ---------------------------------------------------------
# Download once
# ---------------------------------------------------------

if not DATA_PATH.exists():

    print("Downloading evil_numbers...")

    urlretrieve(
        URL,
        DATA_PATH,
    )

    print("Downloaded to:")
    print(DATA_PATH)

else:

    print("Dataset already downloaded:")
    print(DATA_PATH)


# ---------------------------------------------------------
# Read JSONL
# ---------------------------------------------------------

examples = []

with open(
    DATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        examples.append(
            json.loads(line)
        )


print()
print("Number of examples:", len(examples))

print()
print("FIRST EXAMPLE")
print(
    json.dumps(
        examples[0],
        indent=2,
    )
)

print()
print("SECOND EXAMPLE")
print(
    json.dumps(
        examples[1],
        indent=2,
    )
)