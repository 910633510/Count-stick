#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a reviewable ASO diff from the verified live snapshot; no API writes."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
live = json.loads((REPO / 'output/aso-20261027/asc-before.json').read_text())
infos = {x['attributes']['locale']: x for x in next(iter(live['appInfoLocalizations'].values()))}
versions = {x['attributes']['locale']: x for x in next(iter(live['versionLocalizations'].values()))}
chosen = {
    'en-US': {
        'name': 'CountScope: Photo Counter',
        'subtitle': 'Skewers, markers & history',
        'keywords': 'bamboo,wooden,stick,barbecue,hotpot,restaurant,offline,correction,notes,share',
        'promotionalText': 'Count visible skewer ends from a photo. Check and correct each marker, then save or share your result. Includes 4 free automatic counts.',
        'opening': 'Count skewer ends from a photo, check each marker, and save the result. CountScope combines automatic suggestions with manual correction and local history. Take a picture of the exposed ends of a bundle of sticks or skewers, or choose one from Photos.',
    },
    'zh-Hans': {
        'name': '拍照计数 CountScope',
        'subtitle': '竹签识别、手动校对与本地记录',
        'keywords': '串串,烧烤,火锅,餐饮,木签,数签子,点数,相册,标记,分享,备注,离线',
        'promotionalText': '拍下签束端头，查看自动标记，逐点校对后保存或分享结果。免费体验4次自动识别；照片在设备上处理，无需注册账号。',
        'opening': '拍下签束端头，查看自动标记，逐点校对后保存结果。CountScope 将照片识别、手动修正和本地记录放在一起，方便回看和分享。当前自动识别面向一束竹签或木签露出的端头，也可从相册选择照片。',
    },
}
out = {
    'schemaVersion': 1, 'status': 'REVIEWABLE_DRAFT_NOT_APPLIED',
    'createdAtUTC': datetime.now(timezone.utc).isoformat(),
    'appId': '6826601708', 'bundleId': 'x.Count-stick', 'version': '1.0', 'build': '4',
    'versionId': live['versions'][0]['id'], 'selectedBrand': 'CountScope',
    'storeStateAtPreparation': 'WAITING_FOR_REVIEW', 'localizations': {},
    'scopeExclusions': ['Steel-pipe automatic counting', 'colony counting', 'general automatic object counting'],
    'monetizationChanges': False,
    'screenshotDecision': 'Retain authentic Build4 sets: automatic reference outline with review; marker correction; local records and sharing. No new capability claims.',
}
ops = []
for locale, c in chosen.items():
    before = versions[locale]['attributes']
    description = c['opening'] + '\n\n' + before['description'].split('\n\n', 1)[1]
    target_info = {k: c[k] for k in ['name', 'subtitle']}
    target_version = {k: c[k] for k in ['keywords', 'promotionalText']}
    target_version['description'] = description
    checks = {'nameChars': len(c['name']), 'subtitleChars': len(c['subtitle']),
              'keywordUTF8Bytes': len(c['keywords'].encode('utf-8')),
              'promotionalTextChars': len(c['promotionalText']), 'descriptionChars': len(description)}
    assert checks['nameChars'] <= 30 and checks['subtitleChars'] <= 30
    assert checks['keywordUTF8Bytes'] <= 100 and checks['promotionalTextChars'] <= 170 and checks['descriptionChars'] <= 4000
    assert description.split('\n\n', 1)[1] == before['description'].split('\n\n', 1)[1]
    out['localizations'][locale] = {'appInfo': target_info, 'version': target_version, 'validation': checks}
    for kind, old, new in [('appInfoLocalizations', infos[locale], target_info),
                           ('appStoreVersionLocalizations', versions[locale], target_version)]:
        changes = {k: v for k, v in new.items() if old['attributes'].get(k) != v}
        if changes:
            ops.append({'locale': locale, 'method': 'PATCH', 'path': '/v1/' + kind + '/' + old['id'],
                        'before': {k: old['attributes'].get(k) for k in changes},
                        'payload': {'data': {'type': kind, 'id': old['id'], 'attributes': changes}}})
(ROOT / 'metadata-candidate.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
(ROOT / 'metadata-diff.json').write_text(json.dumps({'status': 'DRAFT_NOT_SENT', 'operations': ops,
    'neverIncludes': ['pricing', 'availability', 'build-selection', 'submission-cancellation', 'subscription-products']}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v['validation'] for k, v in out['localizations'].items()}, ensure_ascii=False))
