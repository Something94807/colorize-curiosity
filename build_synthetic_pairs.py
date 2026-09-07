import argparse
import concurrent.futures
import io
import os
import sys

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageFilter, ImageStat

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=24, pool_maxsize=24, max_retries=retries)
session.mount('https://', adapter)
session.mount('http://', adapter)

SOLR_URL = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/select"

def count_existing_files(directory: str, instrument: str) -> int:
    if not os.path.exists(directory):
        return 0
    files = os.listdir(directory)
    if instrument == "mahli":
        return sum(1 for f in files if "MH" in f)
    elif instrument == "mastcam":
        return sum(1 for f in files if "ML" in f or "MR" in f)
    return 0

def fetch_pds_page(instrument: str, start: int, rows: int = 1000) -> list:
    fq_filters = [
        'ATLAS_MISSION_NAME:"mars science laboratory"',
        'IMAGE_TYPE:regular',
        f'ATLAS_INSTRUMENT_NAME:{instrument}',
        'ATLAS_SPACECRAFT_NAME:curiosity'
    ]

    params = {
        "q": "*:*",
        "fq": fq_filters,
        "fl": "ATLAS_BROWSE_URL,ATLAS_PRODUCT_ID,ATLAS_INSTRUMENT_NAME",
        "wt": "json",
        "start": start,
        "rows": rows
    }
    resp = session.get(SOLR_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("response", {}).get("docs", [])

def download_image(url: str) -> Image.Image:
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def is_good_color_image(img: Image.Image) -> bool:
    _, s, v = img.convert("HSV").split()

    if s.getextrema()[1] < 2:
        return False

    v_hist = v.histogram()
    black_pixels = sum(v_hist[:5])
    if (black_pixels / (img.width * img.height)) > 0.10:
        return False

    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    if edge_stat.mean[0] < 10.0:
        return False

    return True

def process_single_doc(doc, args):
    browse_url = doc.get("ATLAS_BROWSE_URL")
    if not browse_url:
        return None, None
    if isinstance(browse_url, list):
        browse_url = browse_url[0]

    browse_url = browse_url.replace("http://", "https://")
    browse_url = browse_url.replace("pdsimg.jpl.nasa.gov", "pds-imaging.jpl.nasa.gov")
    if browse_url.startswith("/"):
        browse_url = "https://pds-imaging.jpl.nasa.gov" + browse_url

    product_id = browse_url.split("/")[-1].split(".")[0]

    if os.path.exists(os.path.join(args.out_dir, "color", f"{product_id}.png")):
        return product_id, "exists"

    inst = str(doc.get("ATLAS_INSTRUMENT_NAME", "")).lower()

    if "mastcam" in inst and not product_id.endswith("DRCX"):
        return product_id, "skipped - Not DRCX"

    if "mahli" in inst and "EDR" in product_id:
        return product_id, "skipped - Raw EDR (Un-debayered)"

    try:
        img = download_image(browse_url)
    except Exception as e:
        return product_id, f"error - {str(e)}"

    if img.width < args.min_width:
        return product_id, "skipped - Too small"

    if max(img.size) > args.max_dim:
        scale = args.max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    if not is_good_color_image(img):
        return product_id, "skipped - Failed quality check"

    r, g, b = img.split()
    gray_arr = (
            np.array(r, dtype=np.float32) * 1.0 +
            np.array(g, dtype=np.float32) * 0.0 +
            np.array(b, dtype=np.float32) * 0.0
    )
    gray_arr = np.clip(gray_arr, 0, 255).astype(np.uint8)
    gray = Image.merge("RGB", (Image.fromarray(gray_arr),) * 3)

    return product_id, (img, gray)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--out-dir", default="mars_dataset")
    ap.add_argument("--min-width", type=int, default=400)
    ap.add_argument("--max-dim", type=int, default=512)
    ap.add_argument("--preview", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    color_dir = os.path.join(args.out_dir, "color")
    gray_dir = os.path.join(args.out_dir, "gray")
    preview_dir = os.path.join(args.out_dir, "preview")

    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(gray_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    target_total = args.count
    mahli_target = target_total // 2
    mastcam_target = target_total - mahli_target

    total_saved = 0

    for instrument, target in [("mahli", mahli_target), ("mastcam", mastcam_target)]:
        existing_count = count_existing_files(color_dir, instrument)
        if existing_count >= target:
            print(f"\n--- {instrument.upper()} complete: Found {existing_count}/{target} existing files. Skipping to next. ---")
            total_saved += target
            continue

        print(f"\n--- Querying PDS Image Atlas for {instrument.upper()} (Target: {target}, Found: {existing_count}) ---")
        saved_inst = 0
        start = 0
        rows = 1000

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            while saved_inst < target:
                docs = fetch_pds_page(instrument, start, rows)
                if not docs:
                    print(f"Reached the end of the PDS search results for {instrument}.")
                    break

                futures = [executor.submit(process_single_doc, doc, args) for doc in docs]

                for future in concurrent.futures.as_completed(futures):
                    if saved_inst >= target:
                        break

                    product_id, result = future.result()

                    if result is None:
                        continue

                    if isinstance(result, str):
                        if result == "exists":
                            saved_inst += 1
                        else:
                            sys.stdout.write(f"\r  Evaluating... {product_id} {result}".ljust(80))
                            sys.stdout.flush()
                        continue

                    img, gray = result
                    fname = f"{product_id}.png"
                    color_path = os.path.join(color_dir, fname)

                    img.save(color_path)
                    gray.save(os.path.join(gray_dir, fname))

                    if total_saved < args.preview:
                        combined = Image.new("RGB", (img.width * 2 + 10, img.height), "white")
                        combined.paste(gray, (0, 0))
                        combined.paste(img, (img.width + 10, 0))
                        combined.save(os.path.join(preview_dir, fname))

                    saved_inst += 1
                    total_saved += 1

                    sys.stdout.write(f"\r  [{saved_inst}/{target} {instrument}] saved {fname}".ljust(80) + "\n")
                    sys.stdout.flush()

                start += rows

    print(f"\nDone. {total_saved} pairs ready in ./{args.out_dir}")

if __name__ == "__main__":
    main()