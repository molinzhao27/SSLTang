"""Public S3 discovery and integrity-checked, resumable OpenNeuro acquisition."""
import hashlib,json,time,urllib.request,urllib.parse,urllib.error,xml.etree.ElementTree as ET
from pathlib import Path
BASE='https://s3.amazonaws.com/openneuro.org'

def discover(dataset):
    result=[];token=None;ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
    while True:
        params={'list-type':'2','prefix':dataset+'/','max-keys':1000}
        if token:params['continuation-token']=token
        with urllib.request.urlopen(BASE+'?'+urllib.parse.urlencode(params),timeout=120) as r:root=ET.fromstring(r.read())
        for item in root.findall('s:Contents',ns):
            result.append({'key':item.find('s:Key',ns).text,'size':int(item.find('s:Size',ns).text),'etag':item.find('s:ETag',ns).text.strip('"')})
        next_token=root.find('s:NextContinuationToken',ns)
        if next_token is None:return result
        token=next_token.text

def download(item,root):
    destination=Path(root)/item['key'];destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.stat().st_size==item['size']:
        return destination
    partial=destination.with_name(destination.name+'.part')
    for attempt in range(6):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>item['size']:partial.unlink();offset=0
            # An interrupted download may already contain the complete object.
            if offset!=item['size'] or not partial.exists():
                headers={'If-Match':'"'+item['etag']+'"'}
                if offset:headers['Range']=f'bytes={offset}-'
                req=urllib.request.Request(BASE+'/'+urllib.parse.quote(item['key'],safe='/'),headers=headers)
                with urllib.request.urlopen(req,timeout=120) as response:
                    if offset and response.status!=206:offset=0
                    with partial.open('ab' if offset else 'wb') as out:
                        while True:
                            chunk=response.read(1024*1024)
                            if not chunk:break
                            out.write(chunk)
            if partial.stat().st_size!=item['size']:raise IOError('size mismatch: '+item['key'])
            # S3 single-part ETags are MD5; multipart ETags are not checksums.
            if '-' not in item['etag']:
                digest=hashlib.md5()
                with partial.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
                if digest.hexdigest()!=item['etag']:partial.unlink();raise IOError('ETag mismatch')
            partial.replace(destination);return destination
        except urllib.error.HTTPError as exc:
            if exc.code in (403,404,412) or attempt==5:raise
            time.sleep(min(2**attempt,30))
        except Exception:
            if attempt==5:raise
            time.sleep(min(2**attempt,30))

def selection(objects,subject,mode):
    dataset='ds003020' if mode=='train' else 'ds004510'; keys={x['key'] for x in objects}
    neural=[x for x in objects if '/preprocessed_data/'+subject+'/' in x['key'] and x['key'].endswith('.hf5') and (mode=='train' or any('/'+c+'/' in x['key'] for c in ['imagined_speech','perceived_movie']))]
    if mode=='train':neural=[x for x in neural if Path(x['key']).stem!='wheretheressmoke']
    needed={x['key'] for x in neural}
    needed.update(k for k in keys if '/sub-'+subject+'/' in k and k.endswith('_bold.json'))
    for x in neural:
        story=Path(x['key']).stem.split('_repeat-')[0]
        candidates=[k for k in keys if '/TextGrids/' in k and Path(k).stem==story]
        needed.update(candidates)
    needed.update(k for k in keys if mode=='transfer' and '/' not in k[len(dataset)+1:] and (k.endswith('_bold.json') or k.endswith('_events.tsv')))
    needed.update(k for k in keys if k.endswith(('README','dataset_description.json','respdict.json','subject_xfms.json')) or (mode=='train' and '/'+subject+'/transforms/' in k and k.endswith('mask_thick.nii.gz')))
    return [x for x in objects if x['key'] in needed]
