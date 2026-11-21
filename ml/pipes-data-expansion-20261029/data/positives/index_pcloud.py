"""Read unchanged public YOLO source; derive provisional HBBs for visual screening only."""
from pathlib import Path
import collections, hashlib, json, math
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
B = ROOT / 'output/pipes-data-expansion-20261029/data/positives'
D = B / 'raw/pcloud-pipes-v2'
R = B / 'review'
R.mkdir(exist_ok=True)
rows = []
for label in sorted(D.glob('*/labels/*.txt')):
    image = label.parent.parent / 'images' / (label.stem + '.jpg')
    assert image.exists()
    with Image.open(image) as im:
        width, height = im.size
    boxes, kinds, issues = [], [], []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.split()]
        assert all(math.isfinite(x) for x in values)
        if values[0] != 0:
            issues.append('unexpected class')
        if len(values) == 5:
            _, cx, cy, w, h = values
            box = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
            kind = 'author_yolo_bbox'
        elif len(values) >= 7 and len(values) % 2 == 1:
            xs, ys = values[1::2], values[2::2]
            box = [min(xs)*width, min(ys)*height, max(xs)*width, max(ys)*height]
            kind = 'derived_HBB_from_author_polygon'
        else:
            issues.append('unsupported annotation row length')
            continue
        if box[2] <= box[0] or box[3] <= box[1] or box[0] < -.001 or box[1] < -.001 or box[2] > width+.001 or box[3] > height+.001:
            issues.append('invalid or out of bounds author region')
        boxes.append(box)
        kinds.append(kind)
    source_member = str(image.relative_to(D))
    rows.append({'id': 'rfv2_'+hashlib.sha256(source_member.encode()).hexdigest()[:12],
                 'imagePath': str(image.relative_to(ROOT)), 'sourceMember': source_member,
                 'labelPath': str(label.relative_to(ROOT)),
                 'sourceLabelSHA256': hashlib.sha256(label.read_bytes()).hexdigest(),
                 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'width': width, 'height': height,
                 'authorSplit': image.relative_to(D).parts[0], 'sourceStem': label.stem.split('.rf.')[0],
                 'count': len(boxes), 'xyxy': boxes, 'annotationKinds': kinds, 'technicalIssues': issues})
(B/'sources/pcloud-pipes-v2-technical-index.json').write_text(json.dumps({'images': rows}, indent=2)+'\n')
groups = collections.defaultdict(list)
for row in rows:
    if 1 <= row['count'] <= 20 and not row['technicalIssues']:
        groups[row['sourceStem']].append(row)
candidates = []
for stem, group in sorted(groups.items()):
    row = sorted(group, key=lambda r: (r['authorSplit'] == 'train', r['sha256']))[0]
    candidates.append(dict(row, number=len(candidates)+1, sourceStemVariants=len(group)))
(B/'sources/pcloud-pipes-v2-lowcount-screen-index.json').write_text(json.dumps(candidates, indent=2)+'\n')
for start in range(0, len(candidates), 30):
    sheet = Image.new('RGB', (1500, 1200), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for j, row in enumerate(candidates[start:start+30]):
        with Image.open(ROOT/row['imagePath']) as im:
            im = im.convert('RGB'); im.thumbnail((295, 165))
            x, y = j%5*300, j//5*200
            sheet.paste(im, (x, y))
            draw.text((x+3, y+167), f"{row['number']:03d} GT{row['count']} "+row['sourceStem'][:24], fill='black')
    sheet.save(R/f'rfv2-lowcount-{start//30:02d}.jpg', quality=93)
print('ROWS', len(rows), 'LOWCOUNT_STEMS', len(candidates), 'SHEETS', (len(candidates)+29)//30,
      'TECH_ISSUES', sum(bool(r['technicalIssues']) for r in rows), flush=True)
