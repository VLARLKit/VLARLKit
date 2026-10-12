import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

pytest.importorskip("openpi")
from openpi import transforms
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared import normalize
from transformers import GemmaConfig, GemmaForCausalLM, GemmaModel

from vlarlkit.models.modules.qrflow_critic import ActionInjectedEnsembleQ
from vlarlkit.policies.qrflow import QRFlowPolicy
from vlarlkit_openpi.dataconfigs.libero_dataconfig import LiberoInputs, LiberoOutputs
from vlarlkit_openpi.openpi_for_qrflow import OpenPi0ForQRFlow, OpenPi0QRFlowConfig, qrflow_attention


def _gemma_config(adaptive=False):
    return GemmaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                       vocab_size=20, use_adarms=adaptive, adarms_cond_dim=16 if adaptive else None)


class TinyPair(PaliGemmaWithExpertModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.paligemma = nn.Module()
        self.paligemma.language_model = GemmaModel(_gemma_config()).float()
        self.paligemma.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
        self.gemma_expert = GemmaForCausalLM(_gemma_config(True)).float()
        self.gemma_expert.model.embed_tokens = None

    def embed_image(self, image):
        return image.mean((1, 2, 3))[:, None, None].expand(-1, 1, 16)


class Tokenize:
    def __call__(self, data):
        data.pop("prompt")
        data["tokenized_prompt"] = np.array([1, 2, 3, 0], dtype=np.int64)
        data["tokenized_prompt_mask"] = np.array([True, True, True, False])
        return data


@pytest.fixture
def norm_stats():
    path = Path("/network/scratch/s/sunyi/rlinf_sft_models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero")
    if path.exists():
        return normalize.load(path)
    return {key: normalize.NormStats(mean=np.zeros(32), std=np.ones(32),
                                     q01=np.concatenate([-np.ones(d), np.zeros(32 - d)]),
                                     q99=np.concatenate([np.ones(d), np.zeros(32 - d)]))
            for key, d in (("state", 8), ("actions", 7))}


@pytest.fixture
def tiny_model(monkeypatch, norm_stats):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def initialize(model, config):
        nn.Module.__init__(model)
        model.config, model.pi05 = config, True
        model.gradient_checkpointing_enabled = False
        model.paligemma_with_expert = TinyPair()
        model.action_in_proj = nn.Linear(32, 16)
        model.action_out_proj = nn.Linear(16, 32)
        model.time_mlp_in = nn.Linear(16, 16)
        model.time_mlp_out = nn.Linear(16, 16)
    monkeypatch.setattr(PI0Pytorch, "__init__", initialize)
    cfg = OpenPi0QRFlowConfig(pi05=True, action_horizon=2, action_chunk=1,
                              num_steps=4, discrete_state_input=False)
    torch.manual_seed(2)
    model = OpenPi0ForQRFlow(cfg)
    # Simulate loading SFT before taking the reference snapshot.
    with torch.no_grad():
        model.action_out_proj.bias.fill_(0.05)
    model.initialize_qrflow()
    model.setup_wrappers(
        transforms=[LiberoInputs(model_type=cfg.model_type), transforms.Normalize(norm_stats, use_quantiles=True),
                    Tokenize(), transforms.PadStatesAndActions(32)],
        output_transforms=[transforms.Unnormalize(norm_stats, use_quantiles=True), LiberoOutputs()],
    )
    normalizer = transforms.Normalize(norm_stats, use_quantiles=True)
    bounds = normalizer({'actions': np.stack([-np.ones(7), np.ones(7)])})['actions']
    model.set_action_bounds(bounds[0, :6], bounds[1, :6])
    return model


def _obs(b=2):
    rng = np.random.RandomState(7)
    return dict(main_images=rng.randint(256, size=(b, 224, 224, 3), dtype=np.uint8),
                wrist_images=rng.randint(256, size=(b, 224, 224, 3), dtype=np.uint8),
                states=rng.randn(b, 8).astype(np.float32), task_descriptions=["move"] * b)


def test_norm_roundtrip_padding_and_generated_actions(tiny_model, norm_stats):
    normalizer = transforms.Normalize(norm_stats, use_quantiles=True)
    inverse = transforms.Unnormalize({"actions": norm_stats["actions"]}, use_quantiles=True)
    raw = np.random.RandomState(9).randn(2, 7).astype(np.float32)
    normalized = normalizer({"actions": raw})["actions"]
    padded = transforms.PadStatesAndActions(32)({"state": np.zeros(8), "actions": normalized})["actions"]
    np.testing.assert_array_equal(padded[:, 7:], 0)
    np.testing.assert_allclose(inverse({"actions": padded})["actions"][:, :7], raw, atol=2e-6)
    actions, info = tiny_model.predict_action_batch(_obs())
    generated = info["forward_inputs"]["normalized_actions"].numpy()
    stats = norm_stats["actions"]
    expected = (generated + 1) / 2 * (stats.q99[:7] - stats.q01[:7] + 1e-6) + stats.q01[:7]
    np.testing.assert_allclose(actions, expected, atol=2e-6)
    assert actions.shape == (2, 1, 7)
    assert info["forward_inputs"]["obs/image/base_0_rgb"].dtype == torch.uint8
    generated = torch.tensor([[[20., -20., 2., 3., 4., 5., 0.4]]])
    padded = np.pad(generated.numpy(), [(0, 0), (0, 0), (0, 25)])
    raw = inverse({'actions': padded})['actions']
    raw[..., :6] = np.clip(raw[..., :6], -1, 1)
    expected = normalizer({'actions': raw})['actions']
    np.testing.assert_allclose(tiny_model.canonical_actions(generated).numpy(), expected[..., :7], atol=2e-6)


def test_cache_reuse_mask_velocity_and_native_suffix_equivalence(tiny_model):
    calls = []
    hook = tiny_model.paligemma_with_expert.paligemma.language_model.register_forward_hook(
        lambda module, args, output: calls.append(output))
    _, info = tiny_model.predict_action_batch(_obs())
    assert len(calls) == 1
    fi = info["forward_inputs"]
    replay = {key[4:]: value for key, value in fi.items() if key.startswith("obs/")}
    context = tiny_model.prefix_from_replay(replay)
    torch.testing.assert_close(context.features, fi["features"])
    padding = torch.tensor([[True, True, False, True, True, True, False]] * 2)
    pooled = (calls[-1].last_hidden_state * padding[..., None]).sum(1) / padding.sum(1, keepdim=True)
    torch.testing.assert_close(context.features[:, :16], pooled)
    before = [(key.clone(), val.clone()) for key, val in context.past_key_values]
    x, times = torch.randn(2, 2, 32), torch.tensor([0.25, 0.5])
    fast = tiny_model.velocity(x, times, context)
    reference = tiny_model.velocity(x, times, context, reference=True)
    torch.testing.assert_close(fast, reference, rtol=0, atol=0)
    # Reconstitute native Pi0 methods with the initial modules saved in state_dict.
    native = OpenPi0ForQRFlow(tiny_model.config)
    state = tiny_model.state_dict()
    remapped = {}
    mapping = {"_decoder.": "paligemma_with_expert.gemma_expert.model.",
               "_action_in.": "action_in_proj.", "_action_out.": "action_out_proj.",
               "_time_in.": "time_mlp_in.", "_time_out.": "time_mlp_out."}
    for key, value in state.items():
        if key.startswith("fast_actor."):
            suffix = key[len("fast_actor."):]
            for source, dest in mapping.items():
                if suffix.startswith(source):
                    remapped[dest + suffix[len(source):]] = value
    native.load_state_dict(remapped, strict=False)
    native.eval()
    # Native get_suffix_out takes the prefix padding, not the already built suffix mask.
    native_hidden = native.get_suffix_out(torch.zeros(2, 32), padding, context.past_key_values, x, 1 - times)
    native_velocity = native.action_out_proj(native_hidden)
    torch.testing.assert_close(fast, -native_velocity, atol=2e-6, rtol=2e-5)
    (fast.square().mean()).backward()
    assert len(calls) == 2  # Both experts/backward reuse the prefix.
    for (key, val), (old_key, old_val) in zip(context.past_key_values, before):
        torch.testing.assert_close(key, old_key, rtol=0, atol=0)
        torch.testing.assert_close(val, old_val, rtol=0, atol=0)
    selected = context.select(torch.tensor([1, 0, 1, 1]))
    repeated = tiny_model.velocity(x[[1, 0, 1, 1]], times[[1, 0, 1, 1]], selected)
    torch.testing.assert_close(repeated, fast[[1, 0, 1, 1]])
    hook.remove()


def test_real_gemma_qrflow_update_keeps_backbone_reference_and_critic_frozen(tiny_model):
    tiny_model.config = replace(tiny_model.config, num_steps=5)
    raw = _obs()
    _, info = tiny_model.predict_action_batch(raw)
    batch = {key: value.cpu().numpy() for key, value in info['forward_inputs'].items() if key.startswith('obs/')}
    observation, _ = tiny_model.prepare_observation(raw)
    context = tiny_model.encode_prefix(observation)
    batch['reference_actions'] = tiny_model.reference_actions(context).cpu().numpy()
    assert batch['reference_actions'].shape == (2, 8, 1, 7)
    critic = ActionInjectedEnsembleQ(feature_dim=context.features.shape[-1], action_dim=7, hidden_dims=[16])
    critic_before = copy.deepcopy(critic.state_dict())
    frozen = {key: value.clone() for key, value in tiny_model.state_dict().items() if not key.startswith('fast_actor.')}
    actor_before = {key: value.clone() for key, value in tiny_model.fast_actor.state_dict().items()}
    cfg = OmegaConf.create(dict(
        algorithm=dict(actor_global_batch_size=2, actor_grad_micro_batch_size=1,
                       qrflow_num_paths=3, beta=0.3, qrflow_cost_clip=10,
                       single_sde_node_seed=11, single_sde_noise_seed=12),
        training=dict(optim=dict(lr=1e-3, clip_grad=1), fsdp_config={}),
        runner=dict(iteration=1),
    ))
    policy = QRFlowPolicy(cfg, tiny_model, critic, 0)
    metrics = policy.run_update(batch)
    assert policy.update_step == 1 and all(np.isfinite(list(metrics.values())))
    for key, value in frozen.items():
        torch.testing.assert_close(tiny_model.state_dict()[key], value, rtol=0, atol=0)
    for key, value in critic_before.items():
        torch.testing.assert_close(critic.state_dict()[key], value, rtol=0, atol=0)
    assert any(not torch.equal(tiny_model.fast_actor.state_dict()[key], value) for key, value in actor_before.items())
    state = policy.state_dict()
    assert {int(item['step']) for item in state['optimizer']['state'].values()} == {1}
