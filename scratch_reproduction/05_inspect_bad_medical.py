from datasets import load_dataset


DATASET_NAME = (
    "geodesic-research/"
    "sfm-emergent-misalignment-training-data"
)

SPLIT = "bad_medical_advice"


print("Loading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split=SPLIT,
)

print()
print("Number of examples:", len(dataset))
print("Columns:", dataset.column_names)

print()
print("FIRST EXAMPLE")
print(dataset[0])