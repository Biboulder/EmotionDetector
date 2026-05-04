import argparse
import random
import uuid
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


def shuffle_folder(folder: Path, seed: int | None) -> int:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"[skip] {folder}: no images")
        return 0

    rng = random.Random(seed)
    order = files[:]
    rng.shuffle(order)

    # Two-pass rename to avoid collisions with existing names.
    tmp_paths = []
    for p in order:
        tmp = p.with_name(f".__shuf_{uuid.uuid4().hex}{p.suffix.lower()}")
        p.rename(tmp)
        tmp_paths.append(tmp)

    width = max(4, len(str(len(tmp_paths))))
    for idx, tmp in enumerate(tmp_paths):
        final = folder / f"img_{idx:0{width}d}{tmp.suffix}"
        tmp.rename(final)

    print(f"[ok]   {folder}: shuffled {len(tmp_paths)} files")
    return len(tmp_paths)


def main():
    parser = argparse.ArgumentParser(description="Shuffle (randomly rename) images inside emotion folders.")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["happy", "sad", "suprised"],
        help="Folders to shuffle (default: happy sad suprised)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--root", default=".", help="Root directory containing the folders")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    total = 0
    for name in args.folders:
        folder = root / name
        if not folder.is_dir():
            print(f"[warn] {folder} does not exist, skipping")
            continue
        total += shuffle_folder(folder, args.seed)

    print(f"Done. {total} files renamed.")


if __name__ == "__main__":
    main()
