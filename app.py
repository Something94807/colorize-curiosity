import gradio as gr
import torch
import torch.nn as nn
import re
import numpy as np
from torchvision import transforms
from PIL import Image

# --- NEURAL NETWORK ARCHITECTURE ---

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

def upconv_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = conv_block(1, 64)
        self.down2 = conv_block(64, 128)
        self.down3 = conv_block(128, 256)
        self.down4 = conv_block(256, 512)
        self.down5 = conv_block(512, 512)
        self.down6 = conv_block(512, 512)

        self.up1 = upconv_block(512, 512, dropout=True)
        self.up2 = upconv_block(1024, 512, dropout=True)
        self.up3 = upconv_block(1024, 256)
        self.up4 = upconv_block(512, 128)
        self.up5 = upconv_block(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 2, kernel_size=3, padding=1)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        out = self.final(torch.cat([u5, d1], dim=1))
        return torch.tanh(out)

def lab_to_rgb(l_tensor, ab_tensor):
    l = ((l_tensor.squeeze(0).cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0).numpy().astype(np.uint8)
    ab = ((ab_tensor.cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0).numpy().astype(np.uint8)
    lab_arr = np.transpose(np.vstack([l[None, ...], ab]), (1, 2, 0))
    return Image.fromarray(lab_arr, mode="LAB").convert("RGB")


# --- NASA .IMG FILE PARSER ---

DTYPE_MAP = {
    ("MSB_INTEGER", 8): ">i1", ("MSB_INTEGER", 16): ">i2", ("MSB_INTEGER", 32): ">i4",
    ("LSB_INTEGER", 8): "<i1", ("LSB_INTEGER", 16): "<i2", ("LSB_INTEGER", 32): "<i4",
    ("MSB_UNSIGNED_INTEGER", 8): ">u1", ("MSB_UNSIGNED_INTEGER", 16): ">u2", ("MSB_UNSIGNED_INTEGER", 32): ">u4",
    ("LSB_UNSIGNED_INTEGER", 8): "<u1", ("LSB_UNSIGNED_INTEGER", 16): "<u2", ("LSB_UNSIGNED_INTEGER", 32): "<u4",
    ("UNSIGNED_INTEGER", 8): "u1", ("IEEE_REAL", 32): ">f4", ("IEEE_REAL", 64): ">f8",
    ("PC_REAL", 32): "<f4", ("PC_REAL", 64): "<f8",
}

class PdsImgError(RuntimeError):
    pass

def read_pds_image(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header_bytes = f.read(200_000)
        text = header_bytes.decode("latin-1", errors="replace")

        rb_m = re.search(r"RECORD_BYTES\s*=\s*(\d+)", text)
        ptr_m = re.search(r"\^IMAGE\s*=\s*(\d+)", text)
        if not rb_m or not ptr_m:
            raise PdsImgError("Could not find RECORD_BYTES / ^IMAGE pointer in the label.")

        record_bytes = int(rb_m.group(1))
        image_record = int(ptr_m.group(1))
        block_m = re.search(r"OBJECT\s*=\s*IMAGE\r?\n(.*?)END_OBJECT\s*=\s*IMAGE\r?\n", text, re.DOTALL)
        block = block_m.group(1) if block_m else text

        def field(pattern, cast=int, default=None):
            m = re.search(pattern, block)
            return cast(m.group(1)) if m else default

        lines = field(r"LINES\s*=\s*(\d+)")
        samples = field(r"LINE_SAMPLES\s*=\s*(\d+)")
        bits = field(r"SAMPLE_BITS\s*=\s*(\d+)")
        sample_type = field(r"SAMPLE_TYPE\s*=\s*(\S+)", cast=str)
        bands = field(r"BANDS\s*=\s*(\d+)", default=1)

        if not all([lines, samples, bits, sample_type]):
            raise PdsImgError("Missing required IMAGE object fields.")

        dtype = DTYPE_MAP.get((sample_type, bits))
        if dtype is None:
            raise PdsImgError(f"Unsupported SAMPLE_TYPE/SAMPLE_BITS combo: {sample_type}/{bits}")

        offset = (image_record - 1) * record_bytes
        n_bytes = lines * samples * bands * (bits // 8)

        f.seek(offset)
        raw = f.read(n_bytes)
        if len(raw) != n_bytes:
            raise PdsImgError("File truncated or label misread.")

    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    arr = arr.reshape(bands, lines, samples)
    if bits == 16 and "INTEGER" in sample_type:
        arr = np.mod(arr, 32768)
    return arr

def pds_to_grayscale_uint8(arr: np.ndarray) -> Image.Image:
    band = arr[0]
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        raise PdsImgError("No finite pixel values found.")
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo: hi = lo + 1.0
    stretched = np.clip((band - lo) / (hi - lo), 0, 1)
    return Image.fromarray((stretched * 255).astype(np.uint8), mode="L")


# --- INFERENCE & WEB APP ---

def colorize(model, device, gray_img: Image.Image, img_size: int) -> Image.Image:
    original_size = gray_img.size
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    x = tf(gray_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_ab = model(x).cpu()

    pred_ab_up = torch.nn.functional.interpolate(
        pred_ab, size=(original_size[1], original_size[0]), mode='bicubic', align_corners=False
    ).squeeze(0)

    l_tensor_full = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])(gray_img)

    return lab_to_rgb(l_tensor_full, pred_ab_up)

device = torch.device("cpu")
model = UNetColorizer().to(device)
checkpoint = torch.load("model_best.pt", map_location=device, weights_only=True)
state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

def process_file(file_obj):
    try:
        arr = read_pds_image(file_obj.name)
        gray_img = pds_to_grayscale_uint8(arr)
        color_img = colorize(model, device, gray_img, img_size=512)
        return gray_img, color_img
    except Exception as e:
        raise gr.Error(f"Failed to process image: {str(e)}")

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload Mars .IMG File", file_types=[".IMG", ".img"]),
    outputs=[
        gr.Image(label="Grayscale Input", type="pil"),
        gr.Image(label="Colorized Output", type="pil")
    ],
    title="Curiosity Rover Image Colorizer",
    description="Upload a raw grayscale Navcam or Hazcam .IMG file to colorize it."
)

if __name__ == "__main__":
    demo.launch()