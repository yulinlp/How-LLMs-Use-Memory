import hashlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITION = "dac_full_vocab"

def install_executor(fixed_direction=False):
    gate_path=ROOT/'scripts/endogenous_memory_gates.py'
    gate_code=gate_path.read_text();anchor='torch.nn.functional.linear(direction, weight)'
    assert gate_code.count(anchor)==1
    gate_code='from dac_projection_safe import project_directions\n'+gate_code.replace(anchor,'project_directions(direction, weight)')
    gate=types.ModuleType('endogenous_memory_gates');gate.__file__=str(gate_path)
    exec(compile(gate_code,str(gate_path),'exec'),gate.__dict__);sys.modules[gate.__name__]=gate
    source=ROOT/'steem_adapt/latent_online_generate.py'
    code=gate.patch_executor(source.read_text(),CONDITION)
    if fixed_direction:
        # Freeze before both DAC's logit probe and the final steering update.
        # The full-context state and DAC strength still update every token.
        code=code.replace('    def apply_attention_state(\n',
            '    frozen_item_directions = None\n    def apply_attention_state(\n',1)
        anchor='        full_mass = []\n'
        assert code.count(anchor)==1
        code=code.replace(anchor,
            '        nonlocal frozen_item_directions\n'
            '        if frozen_item_directions is None:\n'
            '            frozen_item_directions = (current_residuals[:, 1::2] - current_residuals[:, 2::2]).detach().clone()\n'
            '        else:\n'
            '            current_residuals = current_residuals.clone()\n'
            '            current_residuals[:, 1::2] = current_residuals[:, 2::2] + frozen_item_directions\n'+anchor,1)
    executor=types.ModuleType('steem_adapt.latent_online_generate');executor.__file__=str(source);executor.__package__='steem_adapt'
    import steem_adapt
    sys.modules[executor.__name__]=executor
    exec(compile(code,str(source),'exec'),executor.__dict__)
    steem_adapt.latent_online_generate=executor
    return executor,hashlib.sha256(code.encode()).hexdigest(),hashlib.sha256(gate_code.encode()).hexdigest()
