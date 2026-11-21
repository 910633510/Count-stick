"""Bounded classical SHA/RGB/SIFT screen for pretraining smooth-rod additions."""
from pathlib import Path
import concurrent.futures,hashlib,importlib.util,json
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'output/pipes-hardening-20261026/data/negatives'
spec=importlib.util.spec_from_file_location('classical',ROOT/'ml/pipes-20261026/data/group_round_scenes.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def read(p):return json.loads(Path(p).read_text())
def basic(r,tag):
    path=r.get('imagePath') or r.get('path')
    if not path or not (ROOT/path).is_file():return None
    return {'id':tag+str(r.get('id',r.get('number',Path(path).stem))), 'imagePath':path,
            'sha256':r.get('sha256') or hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
            'RGBPixelSHA256':r.get('RGBPixelSHA256'),'pHash64':r.get('pHash64'), 'kind':tag}

def main():
    additions=[basic(r,'') for r in read(OUT/'sources/smooth-addition-download-receipts.json') if r['status'].startswith('DOWNLOADED')]
    comparisons=[]
    old_frozen=[basic(r,'old_frozen:') for r in read(ROOT/'output/datasets/pipe-ends-maco-v1/manifest.json')['images']]
    current=[basic(r,'current:') for r in read(OUT/'snapshots-v1/negative-candidates.json')['images']]
    old_source=[basic(r,'old_source:') for r in read(ROOT/'output/pipes-20261026/data/technical-index.json')['images'] if r['sourceArchive']=='round']
    for path in ['sources/local-candidate-index.json','sources/rebardsc-local-index.json']:
        comparisons += [basic(r,'new_candidate:') for r in read(OUT/path)]
    positives=[]
    for path in (ROOT/'output/pipes-hardening-20261026/data/positives/sources').glob('*-index.json'):
        value=read(path)
        if isinstance(value,list):positives += [basic(r,'new_positive:') for r in value if isinstance(r,dict)]
    comparisons += old_frozen+current+old_source+positives
    by_sha={r['sha256']:r for r in comparisons if r}
    comparisons=list(by_sha.values())
    def hashes(r):
        with Image.open(ROOT/r['imagePath']) as im:
            rgb=im.convert('RGB')
            if not r['RGBPixelSHA256']:r['RGBPixelSHA256']=hashlib.sha256(rgb.tobytes()).hexdigest()
            if not r['pHash64']:
                small=np.asarray(rgb.convert('L').resize((32,32),Image.Resampling.LANCZOS))
                r['pHash64']=f'{int("".join("1" if bit else "0" for bit in c.hash_bits(small)),2):016x}'
        return r
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        additions=list(pool.map(hashes,additions));comparisons=list(pool.map(hashes,comparisons))
    always={r['sha256'] for r in old_frozen+current+positives if r}
    pairs=[];needed={r['id']:r for r in additions};exact=[]
    for a in additions:
        close=sorted(comparisons,key=lambda r:bin(int(a['pHash64'],16)^int(r['pHash64'],16)).count('1'))[:24]
        targets={r['id']:r for r in comparisons if r['sha256'] in always}
        targets.update({r['id']:r for r in close})
        for r in comparisons:
            if a['sha256']==r['sha256'] or a['RGBPixelSHA256']==r['RGBPixelSHA256']:exact.append({'a':a['id'],'b':r['id']})
        for r in targets.values():needed[r['id']]=r;pairs.append((a['id'],r['id']))
    for i,a in enumerate(additions):
        for r in additions[i+1:]:pairs.append((a['id'],r['id']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        features={r['id']:f for r,f in zip(needed.values(),pool.map(c.features,needed.values()))}
    def match(pair):
        a,b=pair;r=c.registration(features[a],features[b])
        if r and r['inliers']>=12 and r['ratio']>=.4:return {'a':a,'b':b,**r}
        return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:flagged=[r for r in pool.map(match,pairs) if r]
    result={'status':'CLASSICAL_ADDITION_SCREEN_COMPLETE','additionCount':len(additions),'comparisonIdentities':len(comparisons),'SIFTPairs':len(pairs),
            'comparisonRule':'All old241/current44/all available new positive source photos; plus24 nearest pHash source-corpus neighbors per addition. SHA/RGB equality against all available comparison identities.',
            'exactSHAOrRGBMatches':exact,'flaggedSIFTPairs':flagged,'additions':additions,'modelInferencePerformed':False,
            'limitation':'No detected correspondence does not prove different acquisition sessions. Manual photo/scene review is required.'}
    (OUT/'smooth-addition-group-screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['additions']}),flush=True)

if __name__=='__main__':main()
