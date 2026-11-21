#!/usr/bin/env python3
"""Read-only ASO evidence collection using the existing local Apple Ads client."""
import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path.home()/'.appleads'))
from appleads_readonly import Config,AppleAdsReadOnly,ClientError,redact_output,POPULARITY_PATH,KEYWORDS_PATH,PHRASES_PATH
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/aso-20261027/apple-ads';OUT.mkdir(parents=True,exist_ok=True)
cfg=Config.load(Path.home()/'.appleads/config.json');client=AppleAdsReadOnly(cfg)
def call(name,method,path,body=None):
 receipt={'retrievedAtUTC':datetime.now(timezone.utc).isoformat(),'method':method,'path':path,'request':body}
 try:receipt['response']=redact_output(client.request(method,path,body));receipt['status']='SUCCESS'
 except ClientError as e:receipt['status']='FAILED';receipt['error']=str(e)
 dest=OUT/(name+'.json');dest.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');dest.chmod(0o600)
 response=receipt.get('response',{}); result=response.get('result')
 print(json.dumps({'name':name,'status':receipt['status'],'error':receipt.get('error'),'resultType':type(result).__name__,'resultKeys':list(result) if isinstance(result,dict) else None,'rows':len(result) if isinstance(result,list) else len(result.get('rows',[])) if isinstance(result,dict) else None},ensure_ascii=False),flush=True)
 return response
me=call('identity-me','GET','/v1/me');acls=call('identity-acls','GET','/v1/acls')
assert str(me['result']['orgId'])==str(cfg.org_id)
assert any(str(x['adAccount']['id'])==str(cfg.ad_account_id) for x in acls['result']['acls'])
account=call('identity-account','GET','/v1/ad-accounts/'+str(cfg.ad_account_id))
print('verified account',json.dumps({k:v for k,v in account.get('result',{}).items() if k in ['id','name','orgId','currency','status']},ensure_ascii=False),flush=True)
terms={'US':['photo counter','object counter','count objects','counting app','counter','tally counter','photo count','count from photo','skewer counter','skewer','count sticks','inventory counter','CountScope','CountLens','TallyLens','CountSnap','SnapCount'], 'CN':['拍照计数','拍照数数','数数','计数器','照片计数','智能计数','自动计数','数签子','数签','竹签','串串','库存盘点','清点','点数','拍照点数','计数相机','数一数']}
for country,seeds in terms.items():
 body={'fields':['rankInGenre','searchPopularityInGenre','searchPopularity1to100','searchPopularity1to5'],'filters':[{'field':'countryOrRegion','operator':'EQUALS','value':country},{'field':'genre','operator':'IN','value':['PRODUCTIVITY_UTILITIES','PHOTO_VIDEO']},{'field':'searchTerm','operator':'IN','value':seeds}], 'timeRange':{'start':'2026-10-18','end':'2026-10-24','granularity':'WEEKLY_SUN_SAT'},'pagination':{'offset':0,'pageSize':1000}}
 call('popularity-'+country,'POST',POPULARITY_PATH,body)
 filters=[{'field':'promotedObjectId','operator':'EQUALS','value':['6826601708']},{'field':'promotedObjectType','operator':'EQUALS','value':['APPSTORE_APP']},{'field':'countriesOrRegions','operator':'IN','value':[country]}]
 call('keywords-discovery-'+country,'POST',KEYWORDS_PATH,{'filters':filters})
 call('keywords-seeds-'+country,'POST',KEYWORDS_PATH,{'filters':filters+[{'field':'terms','operator':'IN','value':seeds}]})
call('phrases-search','POST',PHRASES_PATH,{'filters':[{'field':'queryType','operator':'EQUALS','value':['SEARCH']},{'field':'phrase','operator':'IN','value':sum(terms.values(),[])}]})
