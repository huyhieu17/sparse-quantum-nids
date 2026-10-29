"""Resumable full projected-kernel experiment; failures stop the pipeline."""
import sys,subprocess,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.prepare import DATASETS,prepare,ROOT
from experiments.quantum import run as simulate
from experiments.learn import run as train
from experiments.evaluate import run as evaluate,freeze

def run():
 for ds in DATASETS:prepare(ds)
 simulate(('train','val'),workers=20)
 train(workers=12)
 freeze()
 simulate(('test',),workers=20)
 evaluate()
 subprocess.run([sys.executable,'experiments/report.py'],check=True)
 subprocess.run([sys.executable,'experiments/validate.py'],check=True)
 (ROOT/'COMPLETE').write_text('projected experiment complete; see report for fidelity experiment scope\n')
if __name__=='__main__':run()
