"""Count executed affine synaptic operations per inference decision."""
from collections import defaultdict
from contextlib import AbstractContextManager

import torch
from torch import nn
from torch.nn import functional as F


class SynapticMeter(AbstractContextManager):
    """Count dense MACs and active ACs using declared per-layer input types."""

    def __init__(self, model, spike_inputs=(), event_inputs=()):
        self.model = model
        self.spike_inputs = set(spike_inputs)
        self.event_inputs = set(event_inputs)
        self.handles = []
        self.rows = defaultdict(lambda: {"mac": 0, "ac": 0, "spikes": 0, "sites": 0,
                                         "calls": 0, "event_dense_mac": 0})

    @torch.no_grad()
    def _count(self, name, module, inputs, output):
        source = inputs[0].detach()
        if not isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.GRUCell)):
            raise TypeError(f"Unrecognized affine layer {name}: {type(module)}")
        row = self.rows[name]
        row["calls"] += 1
        if isinstance(module, nn.GRUCell):
            row["mac"] += output.numel() * 3 * (module.input_size + module.hidden_size)
            return
        connections = output.numel() * module.weight[0].numel()
        if name not in self.spike_inputs and name not in self.event_inputs:
            row["mac"] += connections
            return
        if name in self.event_inputs:
            row["event_dense_mac"] += connections
        if not torch.all((source == 0) | (source == 1)):
            raise ValueError(f"Declared spike input {name} contains nonbinary values")
        row["spikes"] += int(torch.count_nonzero(source))
        row["sites"] += source.numel()
        if isinstance(module, nn.Linear):
            row["ac"] += int(torch.count_nonzero(source)) * module.out_features
        else:
            operation = F.conv2d if isinstance(module, nn.Conv2d) else F.conv1d
            # Count every reached connection, with exact stride, padding and groups.
            weights = torch.ones_like(module.weight[:1]).expand(module.groups, -1, *module.weight.shape[2:]).contiguous()
            reached = operation(source.float(), weights.float(), stride=module.stride, padding=module.padding,
                                dilation=module.dilation, groups=module.groups)
            row["ac"] += int(reached.sum(dtype=torch.int64)) * (module.out_channels // module.groups)

    def __enter__(self):
        supported = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.GRUCell)
        registered = {name for name, m in self.model.named_modules() if isinstance(m, supported)}
        unknown = (self.spike_inputs | self.event_inputs) - registered
        if unknown:
            raise ValueError(f"Unknown spike layers: {sorted(unknown)}")
        for name, module in self.model.named_modules():
            if name in registered:
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, output, name=name: self._count(name, mod, args, output)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @staticmethod
    def totals(rows, decisions, policy_hz, *, conservative):
        mac = sum(row["mac"] + (row["event_dense_mac"] if conservative else 0) for row in rows) / decisions
        ac = sum(row["ac"] for row in rows if not (conservative and row["event_dense_mac"])) / decisions
        pj = mac * 4.6 + ac * 0.9
        return {"mac_per_decision": mac, "ac_per_decision": ac, "energy_pj_per_decision": pj,
                "synaptic_power_w": pj * policy_hz * 1e-12}

    def report(self, decisions, policy_hz):
        if decisions <= 0 or policy_hz <= 0:
            raise ValueError("decisions and policy_hz must be positive")
        layers = {}
        for key, values in self.rows.items():
            rate = values["spikes"] / values["sites"] if values["sites"] else None
            layers[key] = {**values, "input_firing_rate": rate}
        conservative = self.totals(layers.values(), decisions, policy_hz, conservative=True)
        strict = self.totals(layers.values(), decisions, policy_hz, conservative=False)
        return {"decisions": decisions, "policy_hz": policy_hz, **conservative, "layers": layers,
                "conservative_first_layer_mac": conservative, "strict_event_ac": strict}
