from pathlib import Path



root = Path("data/raw/iemocap")

print("Root:", root.resolve(), "exists:", root.exists())

if root.exists():

    print("Contents:", [p.name for p in root.iterdir()])

