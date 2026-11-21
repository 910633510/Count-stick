"""Synthetic crop-contract fixtures only; no training/model or heldout content."""
import copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import pipe_hardening as p

def main():
 out=p.OUT/'crop-contract-smoke';assert not out.exists();out.mkdir(parents=True)
 pixels=np.arange(64*64*3,dtype=np.uint16).reshape(64,64,3).astype(np.uint8);pixels[:4]=0
 parent=out/'synthetic-parent.png';crop=out/'synthetic-crop.png';Image.fromarray(pixels).save(parent);Image.fromarray(pixels[8:28,10:42]).save(crop)
 base={'dataNature':'real_crop','split':'train','count':0,'xyxy':[],'sourceHadBlackMask':True,'hasAnnotationOverlay':False,'isSynthetic':False,'annotationQA':'PASS','cropReview':'PASS','groupId':'synthetic-parent-group','parentGroupId':'synthetic-parent-group','sourceImagePath':str(parent),'sourceImageSHA256':p.sha(parent),'cropRectPixels':[10,8,42,28],'width':32,'height':20}
 p.validate_data_nature(base,crop);checks={'exact_original_pixel_train_negative':'PASS'}
 mutations={'reject_val':{'split':'val'},'reject_test':{'split':'test'},'reject_positive':{'count':1,'xyxy':[[0,0,1,1]]},'reject_parent_hash':{'sourceImageSHA256':'0'*64},'reject_group_drift':{'groupId':'different'},'reject_missing_manual_review':{'cropReview':'PENDING'},'reject_crop_overlay':{'hasAnnotationOverlay':True},'reject_fractional_rect':{'cropRectPixels':[10.,8,42,28]},'reject_outside_rect':{'cropRectPixels':[10,8,100,28]},'reject_alias_mismatch':{'sourceCropXYXY':[11,8,42,28]},'reject_full_masked_parent':{'cropRectPixels':[0,0,64,64]}}
 for name,mutation in mutations.items():
  row={**base,**mutation}
  try:p.validate_data_nature(row,crop)
  except AssertionError:checks[name]='PASS'
  else:raise AssertionError(name)
 changed=pixels[8:28,10:42].copy();changed[0,0,0]^=1;Image.fromarray(changed).save(crop)
 try:p.validate_data_nature(base,crop)
 except AssertionError:checks['reject_single_pixel_edit']='PASS'
 else:raise AssertionError('single pixel was accepted')
 Image.fromarray(pixels[8:28,10:42]).save(crop)
 natural={'dataNature':'real','split':'val','annotationQA':'PASS','isSynthetic':False,'hasAnnotationOverlay':False};p.validate_data_nature(natural,parent);checks['natural_full_photo_val_still_allowed']='PASS'
 report={'status':'PASS_STRICT_TRAIN_ONLY_REAL_CROP_CONTRACT','checks':checks,'commonSHA256':p.sha(p.__file__),'scriptSHA256':p.sha(__file__),'parentSHA256':p.sha(parent),'cropSHA256':p.sha(crop),'fixturesAreSyntheticAndNotTrainingData':True,'trainingStarted':False,'modelInferencePerformed':False,'heldoutContentRead':False};p.save(out/'report.json',report);print(report['status'],len(checks),p.sha(out/'report.json'))
if __name__=='__main__':main()
