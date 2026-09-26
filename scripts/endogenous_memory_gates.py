"""Untrained temporal gates; positive magnitude modulation, never policy signs."""
import torch


class TemporalGate:
    def __init__(self, mode, model=None, alpha=1.):
        if mode not in ('temporal_attention', 'direction_budget', 'fisher_budget', 'dac_full_vocab'):
            raise ValueError(mode)
        self.mode = mode
        self.reference = None
        self.model = model
        self.alpha = alpha

    def __call__(self, attention_density, residuals, coefficients=None):
        if self.mode in ('fisher_budget', 'dac_full_vocab'):
            return self.fisher_gate(residuals, coefficients)
        if self.mode == 'temporal_attention':
            signal = attention_density.float()
        else:
            signal = (residuals[:, 1::2].float() - residuals[:, 2::2].float()).norm(dim=-1)
        signal = torch.nan_to_num(signal, nan=0., posinf=0., neginf=0.).clamp_min(0)
        if self.reference is None:
            self.reference = signal.detach().clone()
        assert signal.shape == self.reference.shape
        denom = signal + self.reference
        numerator = signal if self.mode == 'temporal_attention' else self.reference
        gate = .5 + numerator / denom.clamp_min(1e-12)
        return torch.where(denom > 1e-8, gate, torch.ones_like(gate))

    def fisher_gate(self, residuals, coefficients):
        # Local KL approximation: KL(p(h) || p(h+g*d)) ~= g^2 Var_p(Wd)/2.
        # Per-item budget, not a guarantee on the joint intervention's KL.
        head = self.model.get_output_embeddings()
        weight = getattr(self.model, '_mccs_fp32_lm_head_weight', None)
        if weight is None:
            weight = head.weight.detach().float()
            self.model._mccs_fp32_lm_head_weight = weight
        bias = getattr(head, 'bias', None)
        bias = bias.detach().float() if bias is not None else None
        base = residuals[:, 0].float()
        direction = residuals[:, 1::2].float() - residuals[:, 2::2].float()
        direction = direction * coefficients.unsqueeze(-1)
        base_logits = torch.nn.functional.linear(base, weight, bias)
        prob = base_logits.softmax(-1)
        logits_delta = torch.nn.functional.linear(direction, weight)
        if self.mode == 'dac_full_vocab':
            # DAC-inspired adaptation: full vocabulary rather than nucleus union.
            # Probe strength 2; then gate=min(KL(p_base||p_probe),2).
            log_base = base_logits.log_softmax(-1)
            log_probe = (base_logits[:, None, :] + 2 * logits_delta).log_softmax(-1)
            return (prob[:, None, :] * (log_base[:, None, :]-log_probe)).sum(-1).clamp(0,2)
        logits_delta = logits_delta * self.alpha
        mean = (prob[:, None, :] * logits_delta).sum(-1, keepdim=True)
        variance = (prob[:, None, :] * (logits_delta-mean).square()).sum(-1)
        gate = torch.sqrt(.1 / variance.clamp_min(1e-8)).clamp(.25, 2.)
        return torch.where(variance > 1e-8, gate, torch.ones_like(gate))


def patch_executor(code, mode):
    anchor = '    def apply_attention_state(\n'
    assert code.count(anchor) == 1
    code = code.replace(anchor, f'    temporal_gate = TemporalGate({mode!r}, model=model, alpha=alpha)\n' + anchor)
    anchor = '        full_gate = normalize_attention_gate(torch.stack(full_mass))'
    assert code.count(anchor) == 1
    code = code.replace(anchor,
        '        gate_coefficients = torch.tensor([[action_coefficient(p, coefficients) for p in ps] for ps in policies], device=current_residuals.device, dtype=torch.float32)\n'
        '        full_gate = temporal_gate(torch.stack(full_mass), current_residuals, gate_coefficients)')
    anchor = 'from __future__ import annotations\n'
    assert code.count(anchor) == 1
    return code.replace(anchor, anchor + 'from endogenous_memory_gates import TemporalGate\n')


def self_test():
    residuals = torch.zeros(2, 5, 4)
    for mode in ['temporal_attention', 'direction_budget']:
        gate = TemporalGate(mode)
        residuals[:, 1::2] = 1
        initial = gate(torch.ones(2, 2), residuals)
        assert torch.allclose(initial, torch.ones_like(initial))
        residuals[:, 1] *= 2
        changed = gate(torch.tensor([[2., .5], [2., .5]]), residuals)
        assert torch.isfinite(changed).all() and (changed >= .5).all() and (changed <= 1.5).all()
        assert (changed[:, 0] > 1).all() if mode == 'temporal_attention' else (changed[:, 0] < 1).all()
    single = TemporalGate('temporal_attention')
    single(torch.ones(1, 1), torch.zeros(1, 3, 4))
    assert single(torch.full((1, 1), 2.), torch.zeros(1, 3, 4)).item() > 1
    zero = TemporalGate('direction_budget')
    assert zero(torch.zeros(1, 1), torch.zeros(1, 3, 4)).item() == 1
    class Dummy:
        def get_output_embeddings(self):
            return self.head
    model = Dummy(); model.head = torch.nn.Linear(4, 7)
    gate = TemporalGate('fisher_budget', model)
    output = gate(torch.ones(2,2), torch.randn(2,5,4), torch.ones(2,2))
    assert output.shape == (2,2) and torch.isfinite(output).all()
    assert (output >= .25).all() and (output <= 2).all()
    dac = TemporalGate('dac_full_vocab', model)
    output = dac(torch.ones(2,2), torch.randn(2,5,4), torch.ones(2,2))
    assert torch.isfinite(output).all() and (output >= 0).all() and (output <= 2).all()
    neutral = dac(torch.ones(2,2), torch.randn(2,5,4), torch.zeros(2,2))
    assert torch.allclose(neutral, torch.zeros_like(neutral), atol=1e-6)
    print('PASS: identity at initialization; bounds; finite; single responsiveness; direction damping')


if __name__ == '__main__':
    self_test()
