from PIL import Image

SRC_PATH = "roi_SHARK.png"
DST_PATH = "roi_SHARK_1080p.png"

SCALE = 1.5
TARGET_SIZE = (1920, 1080)

def upgrade_roi(src_path, dst_path):
    img = Image.open(src_path).convert("RGBA")

    # Safety check
    if img.size != (1280, 720):
        print(f"⚠ Warning: unexpected input size {img.size}")

    img_up = img.resize(
        TARGET_SIZE,
        resample=Image.NEAREST  # IMPORTANT
    )

    img_up.save(dst_path)
    print(f"✔ ROI upgraded and saved to: {dst_path}")

if __name__ == "__main__":
    upgrade_roi(SRC_PATH, DST_PATH)
