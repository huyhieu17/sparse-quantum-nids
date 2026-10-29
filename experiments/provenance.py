"""Record code, calibration, dependency and historical-result provenance."""
import sys,json,platform,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data import sha256
from experiments.prepare import ROOT

def run():
 old={str(p):sha256(p) for folder in ('results_all_noise_v10','results_all_noisytrain_v10') for p in Path(folder).glob('*.csv')}
 dest=ROOT/'historical_hashes.json'
 if dest.exists():assert json.loads(dest.read_text())==old
 else:dest.write_text(json.dumps(old,indent=2))
 info=dict(python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','pandas','scikit-learn','qiskit','qiskit-aer')},snapshots={str(p):sha256(p) for p in Path('device_snapshots').glob('*/*.json')},code={str(p):sha256(p) for p in Path('experiments').glob('*.py')},prepared_data={str(p):sha256(p) for p in ROOT.glob('*/data.npz')},protocol_sha256=sha256('experiments/PROTOCOL.md'))
 (ROOT/'provenance.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':run()
