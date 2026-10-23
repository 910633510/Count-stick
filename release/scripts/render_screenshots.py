#!/usr/bin/env python3
"""Compose promotional copy and an ImageGen backdrop around authentic captures.

Requires Pillow. Inputs are per-locale capture manifests, never UI mockups.
Each manifest contains locale, appVersion, build, device, os, appleLanguages,
appleLocale, effectiveLanguage and screenshots [{role, path, sha256}], with
roles result/correction/history. Paths are absolute or relative to the manifest.
Example: python3 render_screenshots.py --manifest sources/en-US/manifest.json
Only the entire screenshot is proportionately resized; it is never cropped,
masked, translated, retouched, or covered. A pixel comparison verifies this.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("result", "correction", "history")
SIZES = {"iphone": (1206, 2622), "ipad": (2064, 2752)}
HEADLINE_LINES = {
    "A photo. A clearer count.": ["A photo.", "A clearer count."],
    "Check it. Correct it.": ["Check it.", "Correct it."],
    "Keep the count handy.": ["Keep the count", "handy."],
    "拍照数签，逐个看清": ["拍照数签，", "逐个看清"],
    "漏了补，错了改": ["漏了补，", "错了改"],
    "数完留一份记录": ["数完", "留一份记录"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, locale, bold=False):
    if locale == "zh-Hans":
        candidates = list(Path("/System/Library/AssetsV2").glob("com_apple_MobileAsset_Font*/*/AssetData/PingFang.ttc"))
        if not candidates:
            raise ValueError("PingFang SC font is required for Chinese promotional text")
        return ImageFont.truetype(str(candidates[0]), size, index=11 if bold else 3)
    return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size, index=0 if bold else 2)


def wrapped(draw, value, face, width, locale):
    tokens = list(value) if locale == "zh-Hans" else value.split(" ")
    separator = "" if locale == "zh-Hans" else " "
    lines, line = [], ""
    for token in tokens:
        candidate = line + separator + token if line else token
        if draw.textlength(candidate, font=face) > width and line:
            lines.append(line)
            line = token
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def read_sources(path, family):
    manifest = json.loads(path.read_text())
    for key in ("locale", "appVersion", "build", "device", "os", "appleLanguages", "appleLocale", "effectiveLanguage"):
        if not manifest.get(key):
            raise ValueError(f"{path}: missing capture evidence {key}")
    locale = manifest["locale"]
    if manifest.get("bundleID", manifest.get("bundleId")) != "x.Count-stick":
        raise ValueError("Capture manifest must identify the new x.Count-stick app")
    if locale not in ("en-US", "zh-Hans"):
        raise ValueError("Only en-US and zh-Hans have approved promotional copy")
    effective = manifest["effectiveLanguage"].lower()
    accepted = {"en-US": {"en", "en-us", "english"},
                "zh-Hans": {"zh", "zh-hans", "simplified chinese", "简体中文"}}
    if effective not in accepted[locale]:
        raise ValueError("Capture's effective UI language does not match the renderer locale")
    supplied = {entry["role"]: entry for entry in manifest["screenshots"]}
    if len(supplied) != len(manifest["screenshots"]):
        raise ValueError("Duplicate screenshot roles in capture manifest")
    if any(role not in supplied for role in ROLES):
        raise ValueError("Each locale needs authentic result, correction and history captures")
    loaded = []
    for role in ROLES:
        entry = supplied[role]
        source_path = Path(entry["path"])
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_path = source_path.resolve()
        if digest(source_path) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {source_path}")
        with Image.open(source_path) as opened:
            screenshot = opened.convert("RGB")
        if family == "iphone" and screenshot.height / screenshot.width < 1.7:
            raise ValueError("An iPhone output cannot use an iPad capture")
        if family == "ipad" and screenshot.height / screenshot.width > 1.6:
            raise ValueError("An iPad output cannot use an iPhone capture")
        loaded.append((role, source_path, screenshot))
    return manifest, loaded


def render(source, copy, locale, family, background, order):
    width, height = SIZES[family]
    canvas = ImageOps.fit(background.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    is_pad = family == "ipad"
    left = 128 if is_pad else 88
    text_width = width - left * 2
    brand_face = font(40 if is_pad else 32, "en-US", True)
    brand = "COUNT STICK"
    draw.text((left, 76), brand, font=brand_face, fill="#DDBB79", stroke_width=0)
    headline_size = 140 if is_pad else 94
    headline_face = font(headline_size, locale, True)
    lines = HEADLINE_LINES.get(copy["headline"], wrapped(draw, copy["headline"], headline_face, text_width, locale))
    while len(lines) > 2:
        headline_size -= 2
        headline_face = font(headline_size, locale, True)
        lines = wrapped(draw, copy["headline"], headline_face, text_width, locale)
    if any(draw.textlength(line, font=headline_face) > text_width for line in lines):
        raise ValueError("Promotional headline exceeds the layout")
    headline_y = 156 if is_pad else 146
    line_height = int(headline_size * 1.18)
    for i, line in enumerate(lines):
        draw.text((left, headline_y + line_height * i), line, font=headline_face, fill="#F8F4E8")
    support_face = font(56 if is_pad else 42, locale)
    support_lines = wrapped(draw, copy["supporting"], support_face, text_width, locale)
    if len(support_lines) > 2:
        raise ValueError("Supporting copy exceeds the layout")
    support_y = 496 if is_pad else 380
    for i, line in enumerate(support_lines):
        draw.text((left, support_y + i * (69 if is_pad else 54)), line, font=support_face, fill="#DEDCCD")
    top = 650 if is_pad else 520
    bottom = height - (95 if is_pad else 90)
    max_width = int(width * (0.84 if is_pad else 0.82))
    scale = min(max_width / source.width, (bottom - top) / source.height)
    screen_size = (round(source.width * scale), round(source.height * scale))
    scaled = source.resize(screen_size, Image.Resampling.LANCZOS)
    x = (width - screen_size[0]) // 2
    y = top
    # Exterior treatment only: shadows and keyline stay outside the screenshot.
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rectangle((x - 14, y + 16, x + screen_size[0] + 14, y + screen_size[1] + 30), fill=(0, 0, 0, 135))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(27))).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x - 7, y - 7, x + screen_size[0] + 6, y + screen_size[1] + 6), fill="#D5C8A7")
    canvas.paste(scaled, (x, y))
    crop = canvas.crop((x, y, x + screen_size[0], y + screen_size[1]))
    if crop.tobytes() != scaled.tobytes():
        raise ValueError("Compositing changed authentic app pixels")
    return canvas, {"screenshotRectangle": [x, y, *screen_size], "proportionalScale": scale,
                    "scaledScreenshotPixelSHA256": hashlib.sha256(scaled.tobytes()).hexdigest(),
                    "unmodifiedScaledScreenshotVerified": True, "headlineLines": lines}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--family", choices=SIZES, default="iphone")
    parser.add_argument("--background", type=Path, default=ROOT / "release/artwork/marketing/background.png")
    parser.add_argument("--output", type=Path, default=ROOT / "release/screenshots/final")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "release/metadata-draft.json").read_text())
    background = Image.open(args.background).convert("RGB")
    for manifest_path in args.manifest:
        manifest, sources = read_sources(manifest_path.resolve(), args.family)
        locale = manifest["locale"]
        output = args.output / args.family / locale
        output.mkdir(parents=True, exist_ok=True)
        receipt = {"createdAtUTC": datetime.now(timezone.utc).isoformat(), "locale": locale,
                   "appVersion": manifest["appVersion"], "build": manifest["build"],
                   "sourceManifest": str(manifest_path.resolve()), "sourceManifestSHA256": digest(manifest_path),
                   "background": str(args.background.resolve()), "backgroundSHA256": digest(args.background),
                   "sourceEvidence": {key: manifest[key] for key in ("device", "os", "appleLanguages", "appleLocale", "effectiveLanguage")},
                   "rendering": "Entire authentic screenshot proportionately scaled; no UI edits or masks.",
                   "dimensionsReference": "https://developer.apple.com/help/app-store-connect/reference/app-information/screenshot-specifications/",
                   "visualReview": "PENDING", "screenshots": []}
        receipt["sourceEvidence"]["configuration"] = manifest.get("configuration", "not recorded")
        receipt["releaseCaptureGate"] = "RELEASE_SOURCE_RECORDED" if manifest.get("configuration", "").lower().startswith("release") else "PENDING: source capture configuration is not Release"
        images = []
        for index, (role, path, source) in enumerate(sources, start=1):
            copy = metadata["localizations"][locale]["screenshots"][index - 1]
            result, evidence = render(source, copy, locale, args.family, background, index)
            destination = output / f"{index:02d}-{role}.png"
            result.save(destination, "PNG", optimize=True)
            receipt["screenshots"].append({"role": role, "source": str(path), "sourceSHA256": digest(path),
                  "path": str(destination.resolve()), "sha256": digest(destination), "pixels": list(result.size),
                  "headline": copy["headline"], "supporting": copy["supporting"], **evidence})
            images.append(result)
        (output / "render-manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        preview_width = 360
        preview_height = round(images[0].height * preview_width / images[0].width)
        contact = Image.new("RGB", (preview_width * 3, preview_height), "#1E302D")
        for i, image in enumerate(images):
            contact.paste(image.resize((preview_width, preview_height), Image.Resampling.LANCZOS), (i * preview_width, 0))
        contact.save(output / "contact-sheet.jpg", quality=92)
        print(output)


if __name__ == "__main__":
    main()
