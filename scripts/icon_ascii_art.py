from PIL import Image

ICON_PATH = "icon.png"

# Bold, blocky ramp (dark → light)
ASCII_CHARS = " █▓▒░"

PIXEL_WIDTH = 2   # <-- increase to 3 if you want it THICC

def image_to_ascii(path):
    img = Image.open(path).convert("L")

    w, h = img.size
    pixels = img.load()

    ramp_len = len(ASCII_CHARS) - 1

    for y in range(h):
        line = []
        for x in range(w):
            val = pixels[x, y]  # 0–255
            idx = int(val / 255 * ramp_len)
            ch = ASCII_CHARS[idx]
            line.append(ch * PIXEL_WIDTH)  # 👈 fill instead of spacing
        print("".join(line))

if __name__ == "__main__":
    image_to_ascii(ICON_PATH)
