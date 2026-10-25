#!/usr/bin/env python3
"""Fetch Apple's current public OpenAPI schema for release-helper validation."""
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://developer.apple.com/sample-code/app-store-connect/app-store-connect-openapi-specification.zip'
ROOT = Path(__file__).resolve().parents[2]
with urlopen(URL, timeout=60) as response:
    archive = ZipFile(io.BytesIO(response.read()))
    files = [name for name in archive.namelist() if name.endswith('.json') and not name.startswith('__MACOSX/')]
    if len(files) != 1:
        raise SystemExit('Unexpected Apple schema archive; inspect before using it')
    schema = json.loads(archive.read(files[0]))
if not schema.get('openapi') or 'schemas' not in schema.get('components', {}):
    raise SystemExit('Response is not an OpenAPI schema')
path = ROOT / 'output/apple-openapi.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(schema))
print('Saved Apple OpenAPI', schema['info']['version'], 'to', path)
