import re


def bounded_attention_gate(mass):
    import torch
    if mass.ndim != 2 or mass.shape[1] == 0:
        raise ValueError('Expected nonempty [batch,memory] attention densities')
    safe = torch.nan_to_num(mass.float(), nan=0., posinf=0., neginf=0.).clamp_min(0)
    mean = safe.mean(dim=1, keepdim=True)
    gate = 0.5 + safe / (safe + mean).clamp_min(1e-12)
    return torch.where(mean > 1e-8, gate, torch.ones_like(gate))


SINGLE = re.compile(r'([^\s])\1{19,}')
PERIODIC = re.compile(r'(.{2,30}?)\1{9,}')


def repetition(text):
    for pattern in [SINGLE, PERIODIC]:
        for match in pattern.finditer(text):
            unit = match.group(1)
            if any(c.isalnum() for c in unit) and (len(unit) == 1 or len(match.group()) >= 40):
                return dict(unit=unit, characters=len(match.group()), offset=match.start())
    return None
