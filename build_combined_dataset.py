"""
Builds emotion_dataset_combined/ by:
  1. Copying AffectNet images into train and val only (test is reserved for ESP)
  2. Copying all ESP camera images into test (simulates real deployment conditions)
  3. Applying offline augmentation (AUG_PER_IMAGE variants per training image)

ESP folders:  happy/, sad/, suprised/  (suprised → surprise)
Output:       emotion_dataset_combined/

Usage:
    python build_combined_dataset.py
"""

import os
import random
import shutil
import math
from PIL import Image, ImageEnhance, ImageFilter

SEED = 42
random.seed(SEED)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DS      = os.path.join(SCRIPT_DIR, 'emotion_dataset')
OUT_DS      = os.path.join(SCRIPT_DIR, 'emotion_dataset_combined')
CLASSES     = ['happy', 'sad', 'surprise']
SPLITS      = ['train', 'val', 'test']

# ESP source folders (suprised is a typo → maps to surprise)
ESP_SOURCES = {
    'happy':    os.path.join(SCRIPT_DIR, 'Pictures/happy'),
    'sad':      os.path.join(SCRIPT_DIR, 'Pictures/sad'),
    'surprise': os.path.join(SCRIPT_DIR, 'Pictures/surprise'),
}

# ESP split — test is ESP-only; AffectNet never enters test
ESP_TEST_COUNT  = 100   # fixed images reserved for test per class
ESP_TRAIN_RATIO = 0.85  # of the remainder: train fraction (rest → val)

AUG_PER_IMAGE = 4   # augmented copies generated per training image


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Applies a random combination of mild transforms suitable for face images.
    Not too subtle (to add diversity) but not so aggressive it breaks expressions.
    """
    w, h = img.size

    # Horizontal flip — 50 % chance
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotation ±12°
    angle = rng.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Zoom / crop 90–110 %
    scale = rng.uniform(0.90, 1.10)
    new_w = int(w * scale)
    new_h = int(h * scale)
    if scale > 1.0:
        # Zoom in: resize larger then crop back to original size
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - w) // 2
        top  = (new_h - h) // 2
        img  = img.crop((left, top, left + w, top + h))
    else:
        # Zoom out: resize smaller then pad (reflect by resizing back)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)

    # Translation ±8 % of image size
    tx = int(rng.uniform(-0.08, 0.08) * w)
    ty = int(rng.uniform(-0.08, 0.08) * h)
    img = img.transform(
        (w, h), Image.AFFINE,
        (1, 0, -tx, 0, 1, -ty),
        resample=Image.BILINEAR,
    )

    # Brightness ±20 %
    factor = rng.uniform(0.80, 1.20)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast ±20 %
    factor = rng.uniform(0.80, 1.20)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Saturation ±15 %
    factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Color(img).enhance(factor)

    # Slight blur — 30 % chance (radius 0.5–1.0)
    if rng.random() < 0.30:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.0)))

    return img


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_images(folder: str, exclude_aug: bool = False) -> list:
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
        and (not exclude_aug or not f.startswith('aug_'))
    )


def copy_image(src_path: str, dst_path: str):
    """Copy image, converting to RGB JPEG to keep consistent format."""
    img = Image.open(src_path).convert('RGB')
    dst_jpg = os.path.splitext(dst_path)[0] + '.jpg'
    img.save(dst_jpg, 'JPEG', quality=95)


def split_esp_images(files: list) -> tuple:
    """Reserve ESP_TEST_COUNT images for test; split the rest 85/15 train/val."""
    random.shuffle(files)
    n_test  = min(ESP_TEST_COUNT, len(files))
    rest    = files[n_test:]
    n_train = math.floor(len(rest) * ESP_TRAIN_RATIO)
    return rest[:n_train], rest[n_train:], files[:n_test]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Wipe and recreate output directory to ensure a clean run
    if os.path.exists(OUT_DS):
        print(f'Removing existing {OUT_DS} ...')
        shutil.rmtree(OUT_DS)
    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DS, split, cls), exist_ok=True)

    stats = {cls: {s: {'orig': 0, 'aug': 0} for s in SPLITS} for cls in CLASSES}

    # ── Pass 1: copy all images ───────────────────────────────────────────────
    for cls in CLASSES:
        print(f'\n── {cls} ──')

        # AffectNet — train and val only (test is reserved for ESP)
        for split in ['train', 'val']:
            src_dir = os.path.join(SRC_DS, split, cls)
            dst_dir = os.path.join(OUT_DS, split, cls)
            files   = list_images(src_dir, exclude_aug=True)
            for fname in files:
                copy_image(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            stats[cls][split]['orig'] += len(files)
            print(f'  AffectNet {split}: {len(files)} images copied')

        # ESP — split across train/val/test
        esp_dir   = ESP_SOURCES[cls]
        esp_files = list_images(esp_dir)
        train_esp, val_esp, test_esp = split_esp_images(esp_files)

        for split, subset in [('train', train_esp), ('val', val_esp), ('test', test_esp)]:
            dst_dir = os.path.join(OUT_DS, split, cls)
            for fname in subset:
                stem = os.path.splitext(fname)[0]
                copy_image(os.path.join(esp_dir, fname), os.path.join(dst_dir, f'esp_{stem}.jpg'))
            stats[cls][split]['orig'] += len(subset)
            print(f'  ESP       {split}: {len(subset)} images copied')

    # ── Equalise training classes ─────────────────────────────────────────────
    # All classes must have the same number of originals before augmentation
    # so that augmentation produces equal totals.
    train_orig_counts = {cls: stats[cls]['train']['orig'] for cls in CLASSES}
    min_train = min(train_orig_counts.values())
    print(f'\nPre-equalisation train counts: { {c: train_orig_counts[c] for c in CLASSES} }')
    print(f'Equalising to {min_train} originals per class')

    for cls in CLASSES:
        train_dir = os.path.join(OUT_DS, 'train', cls)
        all_files = list_images(train_dir)
        excess    = len(all_files) - min_train
        if excess > 0:
            for fname in random.sample(all_files, excess):
                os.remove(os.path.join(train_dir, fname))
            stats[cls]['train']['orig'] = min_train
            print(f'  {cls}: removed {excess} images → {min_train}')

    # ── Pass 2: augmentation (training only) ─────────────────────────────────
    print()
    for cls in CLASSES:
        train_dir  = os.path.join(OUT_DS, 'train', cls)
        orig_files = list_images(train_dir, exclude_aug=True)
        aug_count  = 0
        rng        = random.Random(SEED)

        for fname in orig_files:
            src_path = os.path.join(train_dir, fname)
            img      = Image.open(src_path).convert('RGB')
            stem     = os.path.splitext(fname)[0]
            for i in range(AUG_PER_IMAGE):
                aug_img  = augment(img, rng)
                aug_name = f'aug_{stem}_{i}.jpg'
                aug_img.save(os.path.join(train_dir, aug_name), 'JPEG', quality=92)
                aug_count += 1

        stats[cls]['train']['aug'] = aug_count
        print(f'  {cls} augmented: {aug_count} images ({AUG_PER_IMAGE} per original)')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 55)
    print(f'{"":12}  {"train":>10}  {"val":>6}  {"test":>6}')
    print('=' * 55)
    for cls in CLASSES:
        t = stats[cls]
        train_total = t['train']['orig'] + t['train']['aug']
        print(f'{cls:12}  '
              f'{t["train"]["orig"]} orig + {t["train"]["aug"]} aug = {train_total:>5} total  |  '
              f'val {t["val"]["orig"]:>3}  |  test {t["test"]["orig"]:>3}')
    print(f'\nOutput: {OUT_DS}')


if __name__ == '__main__':
    main()
