#!/usr/bin/env python3
"""Bounded read-only follow-up to sparse Apple Ads results."""
import json,sys,ssl
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,build_opener,ProxyHandler,HTTPSHandler
from urllib.error import HTTPError
sys.path.insert(0,str(Path.home()/'.appleads'))
import appleads_readonly as ads
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/aso-20261027/apple-ads'
cfg=ads.Config.load(Path.home()/'.appleads/config.json');client=ads.AppleAdsReadOnly(cfg)
def call(name,path,body):
 r={'retrievedAtUTC':datetime.now(timezone.utc).isoformat(),'method':'POST','path':path,'request':body}
 try:r['response']=ads.redact_output(client.request('POST',path,body));r['status']='SUCCESS'
 except ads.ClientError as e:r.update(status='FAILED',error=str(e))
 p=OUT/(name+'.json');p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');p.chmod(0o600)
 print(name,json.dumps(r.get('response',{'error':r.get('error')}),ensure_ascii=False),flush=True)
for country,term in [('US','count'),('CN','计数')]:
 body={'fields':list(ads.METRICS),'filters':[{'field':'countryOrRegion','operator':'EQUALS','value':country},{'field':'genre','operator':'IN','value':['PRODUCTIVITY_UTILITIES','PHOTO_VIDEO']},{'field':'searchTerm','operator':'CONTAINS','value':term}], 'timeRange':{'start':'2026-10-18','end':'2026-10-24','granularity':'WEEKLY_SUN_SAT'},'pagination':{'offset':0,'pageSize':1000}}
 call('popularity-related-'+country,ads.POPULARITY_PATH,body)
# Preserve Apple's structured diagnostic without dumping credentials, headers, or full error body.
receipt=json.loads((OUT/'keywords-discovery-US.json').read_text());body=receipt['request'];path=ads.KEYWORDS_PATH
ads.validate_query(path,body);ads.validate_destination('POST',ads.API_ORIGIN+path,auth=False)
req=Request(ads.API_ORIGIN+path,method='POST',data=json.dumps(body).encode(),headers={'Accept':'application/json','Content-Type':'application/json','Authorization':'Bearer '+client._access_token(),'X-AP-Context':'adAccountId='+str(cfg.ad_account_id)})
opener=build_opener(ProxyHandler({}),HTTPSHandler(context=ssl.create_default_context()),ads.NoRedirect())
diag={'retrievedAtUTC':datetime.now(timezone.utc).isoformat(),'path':path}
try:
 with opener.open(req,timeout=30) as res:diag.update(status=res.status,response=ads.redact_output(json.loads(res.read(100000))))
except HTTPError as e:
 raw=e.read(16000);diag['status']=e.code
 try:
  obj=json.loads(raw)
  def safe(x):
   if isinstance(x,dict):return {k:safe(v) for k,v in x.items() if k in {'error','errors','code','message','description','reason','field','errorCode','errorMessage','messageCode','messageDescription'}}
   if isinstance(x,list):return [safe(a) for a in x[:10]]
   if isinstance(x,(str,int)):return x[:1000] if isinstance(x,str) else x
   return None
  diag['diagnostic']=safe(obj)
 except ValueError:diag['diagnostic']='non-JSON error; body withheld'
p=OUT/'keyword-error-diagnostic.json';p.write_text(json.dumps(diag,ensure_ascii=False,indent=2));p.chmod(0o600);print('diagnostic',json.dumps(diag,ensure_ascii=False),flush=True)
