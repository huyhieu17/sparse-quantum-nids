from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
errors=[]
for name,item in manifest['files'].items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:errors.append(name)
if errors:raise SystemExit('FAIL: '+str(errors))
print(f"PASS: {len(manifest['files'])} files match SHA-256")
