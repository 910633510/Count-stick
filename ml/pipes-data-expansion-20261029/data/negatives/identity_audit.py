"""Classical identity-only exclusion audit; no detector, GT access, or test plots.

Root explicitly permits feature-only comparison to the sealed20 solely for
duplicate exclusion. Frozen manifests are projected to identity fields only.
No test pixels are emitted, no count labels are accessed, and no recognition
performance is measured. Every frozen file remains unchanged.
"""
from pathlib import Path
import concurrent.futures,hashlib,importlib.util,json
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'output/pipes-data-expansion-20261029/data/negatives'
spec=importlib.util.spec_from_file_location('classic_registration',ROOT/'ml/pipes-20261026/data/group_round_scenes.py')
methods=importlib.util.module_from_spec(spec);spec.loader.exec_module(methods)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    sources=[ROOT/'output/datasets/pipe-ends-maco-v1/manifest.json',ROOT/'output/datasets/pipe-ends-head-retention-v1/manifest.json']
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    identity_fields=['id','split','groupId','imagePath','sha256','RGBPixelSHA256']
    frozen=[]
    for path in sources:
        doc=read(path)
        frozen.extend([{k:r[k] for k in identity_fields} for r in doc['images']])
    assert len(frozen)==315 and len({r['id'] for r in frozen})==315
    head_identity=read(ROOT/'output/datasets/pipe-ends-head-retention-v1/identity-only.json')
    assert len(head_identity['images'])==74
    witness={r['id']:r for r in head_identity['images']}
    for r in frozen:
        if r['id'] in witness:
            assert all(r[k]==witness[r['id']][k] for k in ['split','groupId','sha256','RGBPixelSHA256'])
    usedgroups={r['groupId'] for r in frozen};knownhash={r['sha256'] for r in frozen}
    families=read(ROOT/'output/pipes-20261026/data/candidate-merge-provisional.json')['globalFamilies']
    number_group={n:f['groupId'] for f in families if f['groupId'] in usedgroups for n in f['allSourceNumbers']}
    aliases=[]
    # Access only identity fields from the old technical inventory, never boxes/counts.
    for r in read(ROOT/'output/pipes-20261026/data/technical-index.json')['images']:
        if r['sourceArchive']!='round':continue
        n=int(str(r['authorImageId']).removeprefix('PipeHBB'))
        if n in number_group and r['sha256'] not in knownhash:
            aliases.append({'id':'source-alias-'+r['id'],'split':'used_family_alias','groupId':number_group[n],
                            'imagePath':r['imagePath'],'sha256':r['sha256'],'RGBPixelSHA256':r['RGBPixelSHA256']})
    old=frozen+aliases
    new=[{**r,'imagePath':r['path']} for r in read(OUT/'sources/download-receipts.json') if r['status'].startswith('DOWNLOADED')]
    for r in new:
        with Image.open(ROOT/r['imagePath']) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB');r['RGBPixelSHA256']=hashlib.sha256(rgb.tobytes()).hexdigest()
    exact=[{'newId':a['id'],'existingId':b['id'],'existingSplit':b['split'],'existingGroup':b['groupId'],
            'sameBytes':a['sha256']==b['sha256'],'sameRGB':a['RGBPixelSHA256']==b['RGBPixelSHA256']}
           for a in new for b in old if a['sha256']==b['sha256'] or a['RGBPixelSHA256']==b['RGBPixelSHA256']]
    oldrows=[{**r,'id':'reference:'+r['id']} for r in old]
    allrows=new+oldrows
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:features=list(pool.map(methods.features,allrows))
    nf,of=features[:len(new)],features[len(new):]
    print('Identity features:',len(nf),'new',len(of),'reference; no test pixel output',flush=True)
    def compare(pair):
        i,j,isold=pair;a=nf[i];b=of[j] if isold else nf[j]
        hd=min(int(np.count_nonzero(a['canonical']!=b['canonical'])),int(np.count_nonzero(a['ordinary']!=b['ordinary'])))
        r=methods.registration(a,b) or {'matches':0,'inliers':0,'ratio':0.,'coverageA':0.,'coverageB':0.,'flip':False}
        if hd<=12:
            other=methods.registration(a,b,True)
            if other and other['inliers']>r['inliers']:r=other
        strong=r['inliers']>=20 and r['ratio']>=.6 and min(r['coverageA'],r['coverageB'])>=.04
        suspicious=strong or hd<=8 or (r['inliers']>=12 and r['ratio']>=.4)
        return {'a':new[i]['id'],'b':old[j]['id'] if isold else new[j]['id'],'existing':isold,
                'referenceSplit':old[j]['split'] if isold else None,'referenceGroup':old[j]['groupId'] if isold else None,
                'hashDistance':hd,**r,'strong':strong,'suspicious':suspicious}
    pairs=[(i,j,True) for i in range(len(new)) for j in range(len(old))]
    pairs += [(i,j,False) for i in range(len(new)) for j in range(i+1,len(new))]
    flagged=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for count,r in enumerate(pool.map(compare,pairs),1):
            if r['suspicious']:flagged.append(r)
            if count%2500==0:print('Classical pairs',count,'/',len(pairs),flush=True)
    assert all(sha(ROOT/rel)==digest for rel,digest in source_hashes.items())
    result={'status':'CLASSICAL_IDENTITY_SCREEN_COMPLETE_PENDING_MANUAL_ADMISSION','auditSourceSHA256':sha(__file__),
            'frozenManifestSHA256':source_hashes,'frozenViews':315,'knownUsedSourceFamilyAliases':len(aliases),'newCandidates':len(new),
            'pairCount':len(pairs),'exactByteOrRGBMatches':exact,'flaggedPairs':flagged,
            'newImageIdentities':[{k:r[k] for k in ['id','imagePath','sha256','RGBPixelSHA256']} for r in new],
            'sealed20Use':'Explicit root-authorized classical feature-only duplicate exclusion. No test images displayed/exported, no GT fields accessed, no detector or error analysis; all test rows unchanged.',
            'method':'All new x frozen315+known used family aliases, and all new pairs; SIFT500/RANSAC and original/canonical pHash; mirror check when hash<=12.',
            'recognitionModelInferencePerformed':False,'groundTruthUsed':False,'testPixelsEmitted':False,'frozenFilesModified':False,
            'limitation':'Unmatched features cannot prove distinct physical capture sessions. Conservative manual source-family grouping is required.'}
    path=OUT/'identity-audit.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',len(exact),'exact',len(flagged),'flagged',sha(path),flush=True)

if __name__=='__main__':main()
