"""Register this app and reuse existing local distribution certificate via ASC API."""
import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path.home()/'.appstoreconnect'))
from asc_api import request
OUT=Path.home()/'.codex/app-store-connect-release/runs/count-stick-20261022'
OUT.mkdir(parents=True,exist_ok=True)
def save(name,data):
 p=OUT/name
 p.write_text(json.dumps(data,indent=2)+'\n');p.chmod(0o600)
def all_data(path):
 rows=[]
 while path:
  r=request('GET',path);rows.extend(r.get('data',[]));path=r.get('links',{}).get('next')
 return rows
bundles=all_data('/v1/bundleIds?filter[identifier]=x.Count-stick')
if not bundles:
 r=request('POST','/v1/bundleIds',{'data':{'type':'bundleIds','attributes':{'identifier':'x.Count-stick','name':'Count Stick','platform':'IOS'}}})
 bundle=r['data'];save('bundle-created.json',r)
else: bundle=bundles[0]
assert bundle['attributes']['identifier']=='x.Count-stick'
assert bundle['attributes']['seedId']=='5B49H2L7N4'
print('Bundle',bundle['id'],bundle['attributes']['identifier'],bundle['attributes']['seedId'])
identities=subprocess.check_output(['security','find-identity','-v','-p','codesigning'],text=True)
certs=all_data('/v1/certificates?filter[certificateType]=DISTRIBUTION')
matched=[]
for c in certs:
 der=base64.b64decode(c['attributes']['certificateContent'])
 if hashlib.sha1(der).hexdigest().upper() in identities and c['attributes']['displayName']=='XM LLC (US)': matched.append(c)
if len(matched)!=1: raise SystemExit('Expected exactly one matching installed distribution certificate')
cert=matched[0]
print('Existing distribution certificate',cert['id'])
profiles=all_data('/v1/profiles?filter[name]=Count%20Stick%20App%20Store%2020261022')
if profiles:
 profile=profiles[0]
else:
 profile=request('POST','/v1/profiles',{'data':{'type':'profiles','attributes':{'name':'Count Stick App Store 20261022','profileType':'IOS_APP_STORE'},'relationships':{'bundleId':{'data':{'type':'bundleIds','id':bundle['id']}},'certificates':{'data':[{'type':'certificates','id':cert['id']}]}}}})['data']
assert profile['attributes']['profileState']=='ACTIVE'
profile_bundle=request('GET',f"/v1/profiles/{profile['id']}/bundleId")['data']
assert profile_bundle['id']==bundle['id'], 'Existing named profile targets another bundle'
profile_certs=all_data(f"/v1/profiles/{profile['id']}/certificates")
assert cert['id'] in [c['id'] for c in profile_certs], 'Profile does not contain local distribution identity'
uuid=profile['attributes']['uuid']
dest=Path.home()/'Library/Developer/Xcode/UserData/Provisioning Profiles'/f'{uuid}.mobileprovision'
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_bytes(base64.b64decode(profile['attributes']['profileContent']));dest.chmod(0o600)
receipt={'bundle_id':'x.Count-stick','bundle_resource_id':bundle['id'],'team_id':bundle['attributes']['seedId'],'certificate_id':cert['id'],'profile_id':profile['id'],'profile_uuid':uuid,'profile_name':profile['attributes']['name'],'profile_state':profile['attributes']['profileState']}
save('signing-ready.json',receipt)
print(json.dumps(receipt,indent=2))
