"""Classical duplicate/scene screen for new negative candidates. No learned model."""
from pathlib import Path
import concurrent.futures, hashlib, importlib.util, json
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'output/pipes-hardening-20261026/data/negatives'
OLD = ROOT / 'output/datasets/pipe-ends-maco-v1/manifest.json'
spec = importlib.util.spec_from_file_location('classical_scene_registration', ROOT/'ml/pipes-20261026/data/group_round_scenes.py')
methods = importlib.util.module_from_spec(spec); spec.loader.exec_module(methods)


def main():
    new = json.loads((OUT/'sources/local-candidate-index.json').read_text())
    new += json.loads((OUT/'sources/rebardsc-local-index.json').read_text())
    new += [r for r in json.loads((OUT/'sources/web-download-receipts.json').read_text()) if r['status'].startswith('DOWNLOADED')]
    for r in new:
        r['imagePath'] = r['path']; r['kind'] = 'new'; r['split'] = None
    old = json.loads(OLD.read_text())['images']
    old = [{**r, 'id': 'old-'+r['id'], 'kind':'old_frozen'} for r in old]
    old_by_sha = {r['sha256']:r for r in old}
    families = json.loads((ROOT/'output/pipes-20261026/data/candidate-merge-provisional.json').read_text())['globalFamilies']
    number_group = {n:f['groupId'] for f in families for n in f['allSourceNumbers']}
    group_split = {r['groupId']:r['split'] for r in old}
    for r in json.loads((ROOT/'output/pipes-20261026/data/technical-index.json').read_text())['images']:
        if r['sourceArchive'] != 'round' or r['sha256'] in old_by_sha: continue
        group = number_group.get(int(str(r['authorImageId']).removeprefix('PipeHBB')))
        old.append({**r, 'id':'old-source-'+r['id'], 'kind':'old_source_family', 'groupId':group, 'split':group_split.get(group)})
    rows = new + old; n = len(new)
    features = [None] * len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(methods.features,r):i for i,r in enumerate(rows)}
        for done,f in enumerate(concurrent.futures.as_completed(futures),1):
            features[futures[f]] = f.result()
            if done % 200 == 0: print('features',done,'/',len(rows),flush=True)
    canonical = np.stack([r['canonical'] for r in features]); ordinary = np.stack([r['ordinary'] for r in features])
    dist = np.minimum(cdist(canonical[:n],canonical,metric='hamming'), cdist(ordinary[:n],ordinary,metric='hamming'))*64
    pairs=set(); old_neg=[j for j in range(n,len(rows)) if rows[j]['kind']=='old_frozen' and rows[j]['count']==0]
    for i in range(n):
        dist[i,i]=1000
        for region in [range(n),range(n,len(rows))]:
            closest=sorted(region,key=lambda j:dist[i,j])[:12]
            for j in closest:
                if dist[i,j]<=24:pairs.add(tuple(sorted((i,j))))
        for j in np.where(dist[i]<=10)[0]:pairs.add(tuple(sorted((i,int(j)))))
        for j in old_neg:pairs.add((i,j))
    print('pairs',len(pairs),flush=True)
    edges=[]; suspicious=[]; exact=[]
    for i in range(n):
        for j in range(i+1,len(rows)):
            if rows[i]['sha256']==rows[j]['sha256']:exact.append({'a':rows[i]['id'],'b':rows[j]['id'],'kind':'byte_identical'})
    def match(pair):
        i,j=pair;result=methods.registration(features[i],features[j])
        result=result or {'matches':0,'inliers':0,'ratio':0,'coverageA':0,'coverageB':0,'flip':False}
        def passed(x):return x['inliers']>=20 and x['ratio']>=.6 and min(x['coverageA'],x['coverageB'])>=.04
        if not passed(result) and dist[i,j]<=10:
            other=methods.registration(features[i],features[j],True)
            if other and other['inliers']>result['inliers']:result=other
        return i,j,result,passed(result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for done,(i,j,result,passed) in enumerate(pool.map(match,sorted(pairs)),1):
            row={'a':rows[i]['id'],'b':rows[j]['id'],'hashDistance':float(dist[i,j]),**result}
            if passed:edges.append(row)
            elif result['inliers']>=12 and result['ratio']>=.4:suspicious.append(row)
            if done%2000==0:print('registered',done,'/',len(pairs),flush=True)
    identity=[{k:r.get(k) for k in ['id','kind','source','imagePath','width','height','sha256','groupId','split']} for r in rows]
    report={'status':'CLASSICAL_SCREEN_REQUIRES_MANUAL_FAMILY_REVIEW','sourceSHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'oldManifestSHA256':hashlib.sha256(OLD.read_bytes()).hexdigest(),'newCandidates':n,'oldFrozenAndSourceViews':len(old),
            'method':'SIFT500/RANSAC, top12 rotational/original pHash neighbors per new/old pool<=24bits; all<=10bits; every old frozen negative checked against each new candidate. No learned features/model outputs.',
            'limitations':'Registration cannot prove different physical scenes for unmatched images. Cross-publisher duplicate imagery and same-scene context still need manual review.',
            'rows':identity,'exactByteDuplicates':exact,'registeredEdges':edges,'suspiciousPairs':suspicious,'candidatePairs':len(pairs),'modelInferencePerformed':False}
    (OUT/'group-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',len(edges),'edges',len(suspicious),'suspicious',len(exact),'exact',flush=True)


if __name__=='__main__':main()
