import cv2
import numpy as np
from paddleocr import PaddleOCR
import pandas as pd
import pprint
from sklearn.cluster import KMeans
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bgr_to_hex(bgr):
    return '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def luminance(bgr):
    return 0.299 * float(bgr[2]) + 0.587 * float(bgr[1]) + 0.114 * float(bgr[0])


def pure_color_from_cluster(cluster_pixels, text_is_lighter_than_bg):
    brightness = (0.299 * cluster_pixels[:, 2].astype(float)
                + 0.587 * cluster_pixels[:, 1].astype(float)
                + 0.114 * cluster_pixels[:, 0].astype(float))
    n = max(1, len(cluster_pixels) // 10)
    idx = np.argsort(brightness)[-n:] if text_is_lighter_than_bg else np.argsort(brightness)[:n]
    return cluster_pixels[idx].mean(axis=0)


def split_roi_into_2(roi):
    pixels = roi.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(pixels)
    cA, cB = km.cluster_centers_[0], km.cluster_centers_[1]
    pA, pB = pixels[km.labels_ == 0], pixels[km.labels_ == 1]
    if luminance(cA) >= luminance(cB):
        return cA, pA, cB, pB
    else:
        return cB, pB, cA, pA


# ─────────────────────────────────────────────────────────────────────────────
# Global background detection
# ─────────────────────────────────────────────────────────────────────────────

def identify_global_background(img):
    pixels = img.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km.fit(pixels)
    counts  = np.bincount(km.labels_)
    centers = km.cluster_centers_
    bg_idx  = np.argmax(counts)
    bg_bgr  = centers[bg_idx]
    print(f"  Global bg ({counts[bg_idx]/counts.sum()*100:.1f}%): BGR {bg_bgr.astype(int)}  {bgr_to_hex(bg_bgr)}")
    return bg_bgr


# ─────────────────────────────────────────────────────────────────────────────
# Per-word text colour
# ─────────────────────────────────────────────────────────────────────────────

def get_text_color_from_roi(roi, global_bg_bgr, word=""):
    if roi is None or roi.size == 0:
        return None
    h, w = roi.shape[:2]
    if w < 4 or h < 4:
        return None
    if roi.reshape(-1, 3).shape[0] < 6:
        return None

    light_c, light_pix, dark_c, dark_pix = split_roi_into_2(roi)

    dist_light_to_bg = np.linalg.norm(light_c - global_bg_bgr)
    dist_dark_to_bg  = np.linalg.norm(dark_c  - global_bg_bgr)

    if dist_light_to_bg <= dist_dark_to_bg:
        bg_c, txt_pix, text_is_lighter = light_c, dark_pix, False
    else:
        bg_c, txt_pix, text_is_lighter = dark_c, light_pix, True

    colour_dist = np.linalg.norm(light_c - dark_c)
    if colour_dist < 20:
        print(f"    [{word}] ⚠️  Clusters too similar (dist={colour_dist:.1f}) — skipping")
        return None

    if len(txt_pix) == 0:
        return None

    pure_txt_bgr = pure_color_from_cluster(txt_pix, text_is_lighter)
    print(f"    [{word}] bg={bgr_to_hex(bg_c)} | pure_txt={bgr_to_hex(pure_txt_bgr)} | dist={colour_dist:.0f}")
    return pure_txt_bgr


# ─────────────────────────────────────────────────────────────────────────────
# Smart mask builder — automatically picks best strategy per image
# ─────────────────────────────────────────────────────────────────────────────

def best_channel_mask(img):
    """
    For low-contrast images where grayscale Otsu fails, find the single
    channel (R, G, B, or LAB L/A/B) that gives the highest contrast
    between text and background, then threshold it.

    Returns the best binary mask and the strategy name used.
    """
    candidates = {}

    # ── BGR channels ──────────────────────────────────────────────────
    for name, ch in zip(['B','G','R'], cv2.split(img)):
        _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates[name] = (ch, m)

    # ── LAB channels ──────────────────────────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for name, ch in zip(['LAB_L','LAB_A','LAB_B'], cv2.split(lab)):
        _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates[name] = (ch, m)

    # ── Grayscale ─────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates['GRAY'] = (gray, m)

    # ── Pick best: highest inter-class variance (most contrast) ───────
    best_name, best_mask = None, None
    best_score = -1

    for name, (ch, mask) in candidates.items():
        n_white = np.sum(mask == 255)
        n_black = np.sum(mask == 0)
        total   = n_white + n_black

        # Skip degenerate splits (one class < 3% or > 97%)
        ratio = min(n_white, n_black) / total
        if ratio < 0.03:
            continue

        # Inter-class variance = w0*w1*(mean0-mean1)^2  (Fisher criterion)
        w0 = n_black / total
        w1 = n_white / total
        mean0 = ch[mask == 0].mean()   if n_black > 0 else 0
        mean1 = ch[mask == 255].mean() if n_white > 0 else 0
        score = w0 * w1 * (mean0 - mean1) ** 2

        if score > best_score:
            best_score = score
            best_name  = name
            best_mask  = mask

    print(f"  Best mask channel: {best_name} (score={best_score:.1f})")
    return best_name, best_mask


def build_text_mask(img, padding=0):
    """
    Build a tight binary mask covering all text pixels.

    Strategy:
      1. Find the best color channel for separating text from background
         (handles both high-contrast and low-contrast images automatically).
      2. Ensure text = white in mask (minority class).
      3. Morphological close  → fill gaps inside letter strokes.
      4. Dilate (only if padding > 0) → cover anti-aliased edges / add padding.
    """
    strategy, mask = best_channel_mask(img)

    # Ensure text pixels = white (minority)
    n_white = np.sum(mask == 255)
    n_black = np.sum(mask == 0)
    if n_white > n_black:
        mask = cv2.bitwise_not(mask)

    # Close small holes inside letter bodies
    kernel_close = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Optional dilate (only when padding > 0; no padding = use mask as-is)
    if padding > 0:
        kernel_dilate = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel_dilate, iterations=max(1, padding // 2))

    masked_pct = np.sum(mask > 0) / mask.size * 100
    print(f"  Mask covers {np.sum(mask>0)} px ({masked_pct:.1f}% of image)")

    # Safety: if mask covers >60% something went wrong → fallback to OCR boxes
    if masked_pct > 60:
        print(f"  ⚠️  Mask too large ({masked_pct:.1f}%) — falling back to OCR bounding boxes")
        mask = None

    return mask


def build_ocr_box_mask(img, data, padding=0):
    """
    Fallback mask: just fill word bounding boxes from OCR (no padding by default).
    Less precise but always works.
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for _, row in data.iterrows():
        x  = max(int(row['left'])   - padding, 0)
        y  = max(int(row['top'])    - padding, 0)
        x2 = min(int(row['left'])   + int(row['width'])  + padding, img.shape[1])
        y2 = min(int(row['top'])    + int(row['height']) + padding, img.shape[0])
        cv2.rectangle(mask, (x, y), (x2, y2), 255, -1)
    return mask


def build_combined_word_mask(img, data, padding=0):
    """
    Build a combined mask by filling OCR word bounding boxes (one mask for the whole image).
    This produces word-level masks (boxes) instead of per-character/precise pixel masks.
    """
    return build_ocr_box_mask(img, data, padding=padding)


# ─────────────────────────────────────────────────────────────────────────────
# LaMa inpainting  (falls back to OpenCV TELEA if not installed)
# ─────────────────────────────────────────────────────────────────────────────

def inpaint_with_lama(img_bgr, mask):
    try:
        from simple_lama_inpainting import SimpleLama

        lama     = SimpleLama()
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)
        pil_mask = Image.fromarray(mask)
        result   = lama(pil_img, pil_mask)
        result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        print("  ✅ LaMa inpainting successful")
        return result_bgr

    except ImportError:
        print("  ⚠️  simple-lama-inpainting not installed → pip install simple-lama-inpainting")
        print("  Falling back to OpenCV TELEA...")
        result = cv2.inpaint(img_bgr, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        print("  ✅ OpenCV TELEA inpainting used")
        return result

    except Exception as e:
        print(f"  ❌ LaMa failed: {e} → falling back to OpenCV TELEA")
        return cv2.inpaint(img_bgr, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_and_color(image_path, output_path="output_boxes.png",
                           erased_path="erased.png", use_word_boxes=True, box_padding=0):
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_out = img.copy()
    print(f"Image: {img.shape[1]}x{img.shape[0]} px\n")

    # ── Step 1: Global background ──────────────────────────────────────
    print("[ Step 1 ] Global background detection")
    global_bg_bgr = identify_global_background(img)
    print()

    # ── Step 2: OCR (PaddleOCR) ─────────────────────────────────────────
    print("[ Step 2 ] OCR (PaddleOCR)")
    # instantiate OCR (CPU by default; set use_gpu=True if configured)
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, det_db_unclip_ratio=2.0)
    # try predict() (new API) then fall back to ocr()
    try:
        raw = ocr.predict(image_path)
    except TypeError:
        raw = ocr.ocr(image_path)
    except Exception:
        # final fallback to ocr()
        raw = ocr.ocr(image_path)

    # Debug: print raw PaddleOCR output for inspection
    print("RAW PADDLEOCR OUTPUT (type:", type(raw), "):")
    pprint.pprint(raw)

    # Convert PaddleOCR result to a DataFrame similar to pytesseract.image_to_data
    rows = []
    # Normalize raw output to a list of (box, rec_item) pairs.
    parsed_pairs = []

    # Helper: accept a pair-like item (box, rec) and append
    def _append_pair(box, rec_item):
        if box is None or rec_item is None:
            return
        parsed_pairs.append((box, rec_item))

    # Case A: raw is a list/tuple of pairs OR a list containing a nested list of pairs
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, (list, tuple)):
                continue

            # Detect a direct pair: [box, (text, score)] or (box, (text, score))
            def _is_pair(x):
                if not isinstance(x, (list, tuple)) or len(x) != 2:
                    return False
                box_part, rec_part = x[0], x[1]
                if not isinstance(box_part, (list, tuple)):
                    return False
                # A real box has 4 coordinate points, each being a list/tuple of 2 numbers
                # If box_part's elements are themselves [box, rec] pairs, it's NOT a real box
                if len(box_part) == 4 and all(
                    isinstance(p, (list, tuple)) and len(p) == 2 and
                    all(isinstance(c, (int, float)) for c in p)
                    for p in box_part
                ):
                    # Looks like 4 (x,y) coordinate points → real box
                    pass
                else:
                    return False  # ← This is the key fix
                # rec_part can be tuple/list like (text, score) or dict {'text':..., 'score':...}
                if isinstance(rec_part, dict):
                    return True
                if isinstance(rec_part, (list, tuple)) and len(rec_part) >= 1:
                    return True
                if isinstance(rec_part, str):
                    return True
                return False

            if _is_pair(item):
                _append_pair(item[0], item[1])
                continue

            # Otherwise, item is likely a list of pairs (nested). Scan one level down.
            for sub in item:
                if _is_pair(sub):
                    _append_pair(sub[0], sub[1])

    # Case B: PaddleOCR newer API may return list of dicts per image with dt_boxes and rec_res
    if not parsed_pairs and isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
        image_res = raw[0]
        dt = image_res.get("dt_boxes") or image_res.get("dt_boxes_res") or image_res.get("dt_boxes_list")
        rec = image_res.get("rec_res") or image_res.get("rec_res_list") or image_res.get("rec_responses") or image_res.get("rec_res", [])
        if dt is None:
            for r in raw:
                if isinstance(r, dict) and "dt_boxes" in r and "rec_res" in r:
                    dt = r["dt_boxes"]; rec = r["rec_res"]; break
        if dt is not None and rec is not None:
            for box, rec_item in zip(dt, rec):
                _append_pair(box, rec_item)

    # Case C: raw is a dict directly
    if not parsed_pairs and isinstance(raw, dict):
        dt = raw.get("dt_boxes") or raw.get("dt_boxes_list")
        rec = raw.get("rec_res") or raw.get("rec_res_list")
        if dt is not None and rec is not None:
            for box, rec_item in zip(dt, rec):
                _append_pair(box, rec_item)

    # Build rows from parsed_pairs
    for pair in parsed_pairs:
        try:
            box, rec_item = pair
            # box may be array-like of 4 points or nested; normalize
            if hasattr(box, "tolist"):
                box = box.tolist()
            # rec_item may be tuple/list like (text, score) or dict {'text':..., 'score':...}
            if isinstance(rec_item, (list, tuple)):
                txt = rec_item[0]
                score = float(rec_item[1])
            elif isinstance(rec_item, dict):
                txt = rec_item.get("text", "")
                score = float(rec_item.get("score", 0.0))
            else:
                # fallback: treat rec_item as text
                txt = str(rec_item)
                score = 1.0
        except Exception:
            continue
        # box could be shape (4,2) or nested - ensure list of (x,y)
        try:
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
        except Exception:
            # skip malformed box
            continue
        left = max(min(xs), 0)
        top = max(min(ys), 0)
        width = max(xs) - left
        height = max(ys) - top
        rows.append({"left": left, "top": top, "width": width, "height": height, "text": txt, "conf": score * 100})

    data = pd.DataFrame(rows)
    if data.empty:
        print("Detected 0 word(s)\n")
        return {"text": "", "color": "#000000", "text_regions": []}

    # filter empty text and low confidence (>=40%)
    data = data[data['text'].notna() & (data['text'].str.strip() != "")]
    data = data[data['conf'].astype(float) > 40]
    ocr_text = " ".join(data['text'].astype(str)).strip()
    print(f"Detected {len(data)} word(s): '{ocr_text}'\n")

    # Build text_regions for API/UI overlay (box as 4 points in crop image coords)
    text_regions = []
    for _, row in data.iterrows():
        left = int(row["left"])
        top = int(row["top"])
        w = int(row["width"])
        h = int(row["height"])
        text_regions.append({
            "text": str(row["text"]).strip(),
            "score": round(float(row["conf"]) / 100.0, 4),
            "box": [[left, top], [left + w, top], [left + w, top + h], [left, top + h]],
        })

    # ── Step 3: Per-word colour ────────────────────────────────────────
    print("[ Step 3 ] Per-word colour extraction")
    word_colours = []

    for _, row in data.iterrows():
        word = str(row['text']).strip()
        x  = max(int(row['left']), 0)
        y  = max(int(row['top']),  0)
        x2 = min(x + int(row['width']),  img.shape[1])
        y2 = min(y + int(row['height']), img.shape[0])
        conf = float(row['conf'])

        roi     = img[y:y2, x:x2]
        txt_bgr = get_text_color_from_roi(roi, global_bg_bgr, word=word)

        if txt_bgr is not None:
            word_colours.append(txt_bgr)
            cv2.rectangle(img_out, (x, y), (x2, y2), (0, 200, 0), 2)
            ly = y - 8 if y > 15 else y2 + 14
            cv2.putText(img_out, f"{word} ({conf:.0f}%)", (x, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img_out, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_out, f"{word} (failed)", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # ── Step 4: Aggregate colour ───────────────────────────────────────
    print(f"\n[ Step 4 ] Aggregation ({len(word_colours)} words)")
    if not word_colours:
        pixels    = img.reshape(-1, 3).astype(np.float32)
        km        = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(pixels)
        counts    = np.bincount(km.labels_)
        centers   = km.cluster_centers_
        final_bgr = centers[np.argmin(counts)]
    else:
        final_bgr = np.median(np.array(word_colours), axis=0)

    hex_color = bgr_to_hex(final_bgr)
    print(f"Final BGR : {final_bgr.astype(int)}")
    print(f"Final hex : {hex_color}")

    # ── Step 5: Build ONE combined mask for the entire image ──────────
    # Run channel analysis on the FULL image in a single pass.
    # All text pixels across the whole image go into ONE mask.
    # Inpainting is then called ONCE on that combined mask —
    # LaMa sees the full image context when reconstructing, which is
    # far more natural than inpainting each word box separately.
    print(f"\n[ Step 5 ] Building combined mask (full image, single pass)")
    if use_word_boxes:
        print("  Using word bounding box mask (combined, single pass)")
        mask = build_ocr_box_mask(img, data, padding=box_padding)
    else:
        mask = build_text_mask(img, padding=box_padding)
        # Fallback: OCR bounding boxes — still one combined mask, one inpaint call
        if mask is None:
            print("  Using OCR bounding box mask as fallback (combined, single pass)")
            mask = build_ocr_box_mask(img, data, padding=box_padding)

    mask_path = erased_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_path, mask)
    print(f"  Mask saved: {mask_path}")

    # ── Step 6: Single inpaint call on the full combined mask ─────────
    print(f"\n[ Step 6 ] Inpainting — one call on full image + combined mask")
    erased_img = inpaint_with_lama(img, mask)

    # ── Step 7: Save ──────────────────────────────────────────────────
    sw = tuple(int(c) for c in final_bgr)
    cv2.rectangle(img_out, (5, 5), (180, 42), sw, -1)
    cv2.rectangle(img_out, (5, 5), (180, 42), (160, 160, 160), 1)
    lc = (0, 0, 0) if luminance(final_bgr) > 128 else (255, 255, 255)
    cv2.putText(img_out, f"Text: {hex_color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img_out)
    cv2.imwrite(erased_path, erased_img)

    print(f"\n✅ Annotated  : {output_path}")
    print(f"✅ Mask       : {mask_path}")
    print(f"✅ Text erased: {erased_path}")

    # ── Display in Colab/Jupyter ───────────────────────────────────────
    try:
        from IPython.display import display, Image as IPImage
        print("\n── Annotated ──")
        display(IPImage(filename=output_path))
        print("\n── Mask ──")
        display(IPImage(filename=mask_path))
        print("\n── Text Removed ──")
        display(IPImage(filename=erased_path))
    except ImportError:
        pass

    print(f"\n✅ RESULT → text='{ocr_text}' | color='{hex_color}'")
    return {"text": ocr_text, "color": hex_color, "text_regions": text_regions}


# ── TEST ──────────────────────────────────────────────────────────────────────
# pip install simple-lama-inpainting
# image_path = "croped_images\img8.png"
# result = extract_text_and_color(
#     image_path,
#     output_path="output_boxes.png",
#     erased_path="erased.png",
#     use_word_boxes=True,
#     box_padding=5
# )
# print(f"\nFinal output: {result}")