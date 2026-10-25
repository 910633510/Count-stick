#!/usr/bin/env python3
"""Render frozen manual labels beside existing build 2 predictions; never infer."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output/web-annotation-v2'

def read(path):
    return json.loads(path.read_text())

manifest_path = BASE / 'manifest.json'
manifest = read(manifest_path)
report = read(OUTPUT / 'evaluation/report.json')
predictions = read(OUTPUT / 'evaluation/predictions.json')
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert report['validation']['manifestSha256'] == predictions['manifestSha256'] == manifest_sha
scores = {item['imageId']: item['variants']['candidate'] for item in report['images']}
pred = {item['imageId']: item['variants']['candidate']['predictions'] for item in predictions['images']}
font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font = ImageFont.truetype(font_path, 20)
small = ImageFont.truetype(font_path, 17)
title = ImageFont.truetype(font_path, 27)
names = {
    'root-v2-cotton': '端头商品图',
    'root-v2-relax': '大端头近景（边缘不确定）',
    'reviewer-v2-hktvmall': '带文字近景（边缘不确定）',
    'acq-v2-pospal-clean': '手持签束演示照片',
}
canvas = Image.new('RGB', (1120, 80 + 470 * len(manifest['images']) + 80), '#f3f5f7')
draw = ImageDraw.Draw(canvas)
draw.text((24, 15), '新增网络照片：人工标注与当前 Build 2 检测', font=title, fill='#18222e')
draw.text((24, 52), '左：蓝点为人工标注    右：绿圈匹配 / 红圈误检 / 橙叉漏检 / 灰圈忽略', font=small, fill='#445364')
for row_index, item in enumerate(manifest['images']):
    photo_path = ROOT / item['path']
    annotation_path = ROOT / item['annotationPath']
    assert hashlib.sha256(photo_path.read_bytes()).hexdigest() == item['imageSha256']
    assert hashlib.sha256(annotation_path.read_bytes()).hexdigest() == item['annotationSha256']
    annotation = read(annotation_path)
    metric = scores[item['id']]
    top = 80 + row_index * 470
    draw.text((24, top + 6), names[item['id']], font=font, fill='#18222e')
    eligible = annotation['completeCountVerified']
    truth_text = f"人工可见端头 {len(annotation['points'])}" + ('' if eligible else '（仅确定部分）')
    result_text = f"模型 {metric['predictedWholeImageCount']}，匹配 {metric['TP']} / 误检 {metric['FP']} / 漏检 {metric['FN']}"
    draw.text((24, top + 34), truth_text, font=small, fill='#3d4e60')
    draw.text((580, top + 34), result_text, font=small, fill='#3d4e60')
    raw = Image.open(photo_path).convert('RGB')
    for column in range(2):
        im = ImageOps.contain(raw, (510, 380), Image.Resampling.LANCZOS)
        painter = ImageDraw.Draw(im)
        sx, sy = im.width / raw.width, im.height / raw.height
        def marker(x, y, color, cross=False):
            x, y = x * sx, y * sy
            if cross:
                painter.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                painter.line((x + 4, y - 4, x - 4, y + 4), fill=color, width=2)
            else:
                painter.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        if column == 0:
            for point in annotation['points']:
                marker(point['x'], point['y'], '#008bff')
        else:
            matched = {m['predictionId'] for m in metric['matches']}
            false = set(metric['falsePositivePredictionIds'])
            for point in pred[item['id']]:
                color = '#0aa844' if point['id'] in matched else '#ff2738' if point['id'] in false else '#87939c'
                marker(point['x'], point['y'], color)
            missing = set(metric['missedLabelIds'])
            for point in annotation['points']:
                if point['id'] in missing:
                    marker(point['x'], point['y'], '#ff8c00', cross=True)
        canvas.paste(im, (24 + column * 556 + (510 - im.width) // 2, top + 72))
draw.text((24, canvas.height - 61), '标注先冻结，随后只测固定参数；两张边缘不确定图不计算整图计数误差。', font=small, fill='#445364')
draw.text((24, canvas.height - 34), '3 张商品图 + 1 张 POS 演示图；不是餐后实拍总体准确率，也不是 iPhone 真机验收。', font=small, fill='#445364')
destination = OUTPUT / 'result-review.jpg'
canvas.save(destination, quality=95)
print(destination)
