"""Prepare independently visually reviewed full-photo negatives, train only."""
from pathlib import Path
from collections import Counter
import hashlib,json,datetime
import numpy as np
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'output/pipes-data-expansion-20261029/data/negatives'
ART=Path(__file__).resolve().parent
ROOT_REVIEW=ROOT/'ml/pipes-data-expansion-20261029/data/root-negative-visual-review.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(j,indent=2)+'\n')

def main():
    receipts=read(OUT/'sources/download-receipts.json')
    audit=read(OUT/'identity-audit.json');assert not audit['exactByteOrRGBMatches'] and not audit['flaggedPairs']
    root=read(ROOT_REVIEW);accepted={r['id']:r for r in root['images']}
    assert len(accepted)==20
    reviewed_ids=set(accepted)
    # Every accepted full source photo was also visually inspected by this agent.
    excluded={'expneg-greta','expneg-steelhouse','expneg-leren','expneg-dajin'}
    assert not reviewed_ids&excluded
    evidence=[];images=[]
    heavy_pair={'expneg-ikizler','expneg-steeldynamics'}
    billets={'expneg-bozhong','expneg-ikizler','expneg-jnshagang','expneg-kzmc','expneg-steeldynamics','expneg-zelpo','expneg-zhongji1'}
    for r in sorted(receipts,key=lambda x:x['id']):
        if r['id'] not in reviewed_ids:continue
        source=ROOT/r['path'];assert sha(source)==r['sha256']==accepted[r['id']]['sourceSHA256']
        with Image.open(source) as im:
            original=ImageOps.exif_transpose(im).convert('RGB');source_ori=im.getexif().get(274)
        path=OUT/'images'/(r['id']+'.png');path.parent.mkdir(exist_ok=True)
        assert not path.exists();original.save(path)
        with Image.open(path) as image:
            assert image.mode=='RGB' and image.format=='PNG' and image.getexif().get(274) is None
            assert image.size==original.size and np.array_equal(np.asarray(image),np.asarray(original))
            rgb_sha=hashlib.sha256(image.tobytes()).hexdigest()
        label=OUT/'labels'/(r['id']+'.txt');label.parent.mkdir(exist_ok=True);assert not label.exists();label.write_bytes(b'')
        family='expneg29_heavy_billet_possible_same_family' if r['id'] in heavy_pair else 'expneg29_scene_'+r['id'].removeprefix('expneg-')
        source_id='public-industrial-heavy-billet-possible-same-family' if r['id'] in heavy_pair else 'public-industrial-photo:'+r['sourcePage']
        note='Full source frame: continuous solid cut/ground faces, no discernible pipe opening in foreground or background.'
        if r['id'] in ['expneg-jnshagang','expneg-apollo','expneg-triumph','expneg-zelpo','expneg-zhongji1']:
            note+=' Physical stamping, chalk, paint stripes or attached stock labels are not hollow mouths.'
        if r['id']=='expneg-punit':note+=' Bottom-left camera timestamp is a source publisher overlay, not a counting label.'
        if r['id']=='expneg-liange':note+=' Partial source logo at upper left does not cover a pipe-end annotation or add an artificial scene background.'
        row={'id':r['id'],'split':'train','trainOnly':True,'source':source_id,'groupId':family,
             'imagePath':str(path.relative_to(ROOT)),'width':original.width,'height':original.height,'sha256':sha(path),'RGBPixelSHA256':rgb_sha,
             'labelPath':str(label.relative_to(ROOT)),'labelSHA256':sha(label),'count':0,'xyxy':[],'label':'pipe_end','bucket':'negative',
             'negativeKind':'non_pipe','confuserSubtype':'solid_billet_end' if r['id'] in billets else 'solid_rod_end',
             'annotationQA':'PASS','annotationMethod':'Full-source-image visual zero-pipe-end annotation and independent review by two Codex agents; no detector proposals.',
             'visualReviewers':['colony_papers','root'],'visualNotes':note,'dataNature':'real','isSynthetic':False,'hasAnnotationOverlay':False,
             'hasPublisherOverlay':r['id'] in ['expneg-punit','expneg-liange'],'sourceHadBlackMask':False,'fullSourceFramePreserved':True,
             'sourceImagePath':r['path'],'sourceImageSHA256':r['sha256'],'sourceImageURL':r['imageURL'],'sourcePage':r['sourcePage'],
             'sourceEXIFOrientation':source_ori,'normalization':'Whole-frame EXIF-upright RGB PNG; no crop, resize or synthetic augmentation.',
             'groupBasis':'Conservative potentially shared heavy-billet facility/capture family; same canonical sampling source avoids cross-publisher group inflation.' if r['id'] in heavy_pair else 'Distinct visual scene/source photograph with no detected frozen or new correspondence; capture-session IDs unavailable.',
             'detectorInferencePerformed':False,'modelInferencePerformed':False,'finalTrainingAdmission':False}
        images.append(row);evidence.append({'id':r['id'],'sourceSHA256':r['sha256'],'canonicalSHA256':row['sha256'],'RGBPixelSHA256':rgb_sha,
                                           'status':'PASS_ZERO_PIPE_ENDS_AND_FULL_FRAME_PIXEL_EQUALITY','reviewer':'colony_papers','notes':note})
    assert len(images)==20 and len({r['groupId'] for r in images})==19
    assert len({r['sha256'] for r in images})==20 and len({r['RGBPixelSHA256'] for r in images})==20
    witness_paths=[ROOT/'ml/pipes-20261026/data/frozen-identity-witness.json',ROOT/'output/datasets/pipe-ends-head-retention-v1/identity-only.json']
    old_ids=[]
    for p in witness_paths:old_ids += read(p)['images']
    old_sha={r['sha256'] for r in old_ids};old_rgb={r['RGBPixelSHA256'] for r in read(witness_paths[1])['images']}
    # The old witness has only byte hashes; the frozen315 audit above checked RGB.
    assert not {r['sha256'] for r in images}&old_sha and not {r['RGBPixelSHA256'] for r in images}&old_rgb
    own_review=ART/'own-visual-review.json';save(own_review,{'status':'PASS_20_FULL_SOURCE_PHOTO_NEGATIVES','reviewerType':'Codex agent','images':evidence,
          'excluded':root['excluded'],'countLabelMeaning':'Zero pipe_end only; these photos contain solid rods/billets.','detectorInferencePerformed':False})
    result={'status':'VISUAL_AND_IDENTITY_QA_PASSED_PENDING_ROOT_FREEZE','testContentIncluded':False,'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'images':images,'summary':{'images':20,'groups':19,'train':20,'val':0,'test':0,'pipeEndTargets':0,'dataNature':{'real':20},
                                    'confuserSubtypes':dict(Counter(r['confuserSubtype'] for r in images))},
            'reviewBindings':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT_REVIEW,own_review,OUT/'identity-audit.json',OUT/'sources/download-receipts.json']},
            'sourceFamilyMerges':[{'groupId':'expneg29_heavy_billet_possible_same_family','ids':sorted(heavy_pair),'basis':'Conservative visual family, not a claim of byte-identical photos.'}],
            'sealedDataPolicy':'Existing315rows unchanged. Sealed20 used only in explicitly authorized classical identity exclusion, no GT or detector/error analysis and no test image output.',
            'limits':['Visual review by Codex agents is not forensic authentication of photographic origin or metal grade.','No detected classical correspondence proves absence of every same-site different-view relation.','All new images are train-only candidates; root must merge with positive-source family review and freeze before optimization.']}
    proposal=OUT/'negative-proposal.json';save(proposal,result)
    payload=sum(r.get('bytes',0) for r in receipts)
    qa={'status':'PASS_20_TRAIN_ONLY_NEGATIVES_19_CONSERVATIVE_GROUPS','proposalPath':str(proposal.relative_to(ROOT)),'proposalSHA256':sha(proposal),
        'summary':result['summary'],'downloadedCandidates':len([r for r in receipts if r['status'].startswith('DOWNLOADED')]),'downloadedPayloadBytes':payload,
        'payloadBudgetBytes':400000000,'rootReviewSHA256':sha(ROOT_REVIEW),'ownReviewSHA256':sha(own_review),'identityAuditSHA256':sha(OUT/'identity-audit.json'),
        'identityAudit':{'referenceViews':315,'knownUsedFamilyAliases':60,'candidatePairs':9276,'exactMatches':0,'flaggedMatches':0,'sealed20Purpose':'classical duplicate exclusion only'},
        'fullFrameRGBPixelEquality':True,'allEmptyLabelsVerified':all((ROOT/r['labelPath']).read_bytes()==b'' for r in images),
        'testRowsOrFrozenFilesModified':False,'detectorInferencePerformed':False,'newTrainingPerformed':False,'reviewerType':'Codex agents',
        'collectionsForBothMaterials':[{'url':'https://www.zelpo.eu/continuously-cast-steel-billets/','note':'Primary producer with billet and steel-tube product sections; collections may support separate positive and negative acquisitions.'},
                                      {'url':'https://metalcarecenter.com/products.html','note':'Only solid-stock categories confirmed onthispage; no pipe/tube collection established.'}]}
    save(ART/'dataset-qa.json',qa)
    print(json.dumps({'proposal':str(proposal.relative_to(ROOT)),'proposalSHA256':sha(proposal),'qaSHA256':sha(ART/'dataset-qa.json'),'summary':result['summary']},indent=2),flush=True)

if __name__=='__main__':main()
