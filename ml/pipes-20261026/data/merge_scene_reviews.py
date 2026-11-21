#!/usr/bin/env python3
"""Merge source families and make review assets. This does not assign splits."""
from pathlib import Path
import collections,hashlib,json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-20261026/data';ART=ROOT/'ml/pipes-20261026/data'
def number(identity):return int(identity.split('PipeHBB')[-1])
def main():
    data=json.loads((OUT/'technical-index.json').read_text());rounds=[r for r in data['images'] if r['sourceArchive']=='round'];by={number(r['id']):r for r in rounds};parent={n:n for n in by};joins=[]
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(ids,reason):
        ids=[int(x) for x in ids];assert all(x in parent for x in ids)
        roots=[find(x) for x in ids]
        for x in roots:parent[x]=min(roots)
        joins.append({'numbers':ids,'reason':reason})
    auto=json.loads((OUT/'groups-provisional.json').read_text())
    for g in auto['groups']:
        if len(g['imageIds'])>1:union([number(x) for x in g['imageIds']],'classical_SIFT_registration_component')
    accepted={};rejects={};input_shas={};completed=[]
    for who in ['root','general','colony']:
        p=ART/f'scene-review-{who}.json'
        if not p.exists():continue
        review=json.loads(p.read_text());input_shas[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();completed.append(who)
        for row in review.get('acceptedCandidates',[]):accepted[row['number']]={'reviewer':who,'screenRecord':row}
        for key in ['sameSceneSets','possibleSameSceneSets','conservativePossibleFamilies']:
            for g in review.get(key,[]):union(g['numbers'],who+':'+g.get('reason',g.get('basis',g.get('confidence','manual_family'))))
        for key in ['explicitRejects','quarantine']:
            for row in review.get(key,[]):rejects[row['number']]=row.get('reason','quarantined_by_scene_reviewer')
    union([17,716],'Same-capture edited variants17/716; edit direction unresolved, keep family together even though716excluded')
    rejects[716]='Unknown same-capture edited variant; original/composite direction not proven; exclude from clean-real baseline'
    corrections=json.loads((ART/'annotation-corrections.json').read_text());correction_by_id={x['id']:x for x in corrections['images']}
    groups=collections.defaultdict(list)
    for n in by:groups[find(n)].append(n)
    candidate_rows=[]
    for n,screen in sorted(accepted.items()):
        if n in rejects:continue
        src=by[n];xy=src['xyxy']
        if src['id'] in correction_by_id:xy=list(map(list,dict.fromkeys(map(tuple,xy))))
        candidate_rows.append({**src,'groupId':'maco_round_global_'+f'{find(n):08d}','split':None,'label':'pipe_end','annotationBoundary':'outer_rim_source_HBB_pending_complete_visual_coverage_review','xyxy':xy,'count':len(xy),'sourceAuthorCount':src['count'],'annotationQA':'PENDING_FINAL_VISUAL_QA','dataNature':None,'isSynthetic':None,'hasAnnotationOverlay':None,'sceneReviewer':screen['reviewer'],'correctionApplied':src['id'] in correction_by_id})
    report={'status':'CANDIDATE_MERGE_PENDING_FINAL_VISUAL_QA_NO_SPLIT','reviewersAvailable':completed,'reviewInputSHA256':input_shas,'images':candidate_rows,'candidateCount':len(candidate_rows),'candidateGroups':len({r['groupId'] for r in candidate_rows}),'globalFamilies':[{'groupId':'maco_round_global_'+f'{root:08d}','allSourceNumbers':ns,'candidateNumbers':[number(r['id']) for r in candidate_rows if find(number(r['id']))==root]} for root,ns in sorted(groups.items()) if any(find(number(r['id']))==root for r in candidate_rows)],'unionEvidence':joins,'explicitExclusions':rejects,'sourceAuthorXMLsChanged':False,'modelInferencePerformed':False,'splitAssigned':False,'scope':'Square tube sources are stored separately and not included. Source image-file count is not independent real-scene count.'}
    (OUT/'candidate-merge-provisional.json').write_text(json.dumps(report,indent=2)+'\n')
    full=OUT/'review/candidate-overlays';full.mkdir(exist_ok=True);font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
    for row in candidate_rows:
        target=full/(row['authorImageId']+'.jpg')
        if target.exists():continue
        with Image.open(ROOT/row['imagePath']) as source:
            original=source.convert('RGB');original.thumbnail((1000,760));overlay=original.copy()
            d=ImageDraw.Draw(overlay);sx,sy=original.width/row['width'],original.height/row['height']
            for x1,y1,x2,y2 in row['xyxy']:d.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline='#20ffc4',width=1)
            canvas=Image.new('RGB',(original.width*2,original.height+55),'white');canvas.paste(original,(0,55));canvas.paste(overlay,(original.width,55));text=ImageDraw.Draw(canvas);text.text((8,8),f'{row["authorImageId"]} | source {row["width"]}x{row["height"]} | canonical boxes {row["count"]} | no model',fill='black',font=font);text.text((8,31),'Left: source photo. Right: author HBB with only documented exact-duplicate corrections.',fill='black',font=font);canvas.save(target,quality=95)
    # Side-by-side source variants for classical matches; counts never decide grouping.
    pairs=OUT/'review/registered-families';pairs.mkdir(exist_ok=True)
    for i,g in enumerate([g for g in auto['groups'] if g['images']>1]):
        canvas=Image.new('RGB',(460*len(g['imageIds']),400),'white');d=ImageDraw.Draw(canvas)
        for j,identity in enumerate(g['imageIds']):
            row=by[number(identity)]
            with Image.open(ROOT/row['imagePath']) as im:im=im.convert('RGB');im.thumbnail((450,350))
            canvas.paste(im,(j*460,45));d.text((j*460+5,10),f'{number(identity)} | author GT {row["count"]}',fill='black',font=font)
        canvas.save(pairs/f'family-{i:02d}.jpg',quality=94)
    print(json.dumps({'candidateCount':report['candidateCount'],'candidateGroups':report['candidateGroups'],'reviewers':completed,'status':report['status']}))
if __name__=='__main__':main()
