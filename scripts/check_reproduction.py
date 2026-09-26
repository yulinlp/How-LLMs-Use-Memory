import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from public_files import RUNTIME_FILES

ROOT=Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='duet-clean-checkout-') as directory:
        target=Path(directory)
        for relative in RUNTIME_FILES:
            path=target/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,path)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(target),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1')
        env.pop('RPEVAL_PROMPT_FILE',None)
        commands=[
            [sys.executable,'-m','unittest','discover','-s','tests','-p','test_reproduction.py','-v'],
            [sys.executable,'scripts/smoke_reproduction.py'],
            [sys.executable,'reproduce.py','--help'],
        ]
        for cmd in commands:
            subprocess.run(cmd,cwd=target,env=env,check=True)
        print('PASS: isolated checkout without historical runs, models, private runtime, or anonymous release package.')


if __name__=='__main__':
    main()
