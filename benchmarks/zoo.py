"""Model zoo used by the peakcast benchmarks.

Every entry describes one training workload: how to build the model, how to
make a synthetic batch of a given size, and how to compute the loss. All models
are randomly initialised, so no weights are downloaded.

The zoo is deliberately heterogeneous (convolutional networks, vision
transformers, text encoders, decoders, sequence-to-sequence models, audio,
video, recurrent and custom models) because the calibration of peakcast is
evaluated with whole architecture families held out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Spec:
    name: str
    family: str
    domain: str
    build: Callable[[], nn.Module]
    make_batch: Callable[[int, int, torch.device], Any]
    loss: Callable[[nn.Module, Any], torch.Tensor]
    axis: str  # name of the second size axis
    default: int  # default value of the second axis
    grid: tuple[int, ...] = field(default_factory=tuple)  # other values of the second axis


# ----------------------------------------------------------------------------
# batches and losses
# ----------------------------------------------------------------------------


def _image_batch(classes: int = 1000):
    def make(b: int, r: int, dev: torch.device):
        return torch.randn(b, 3, r, r, device=dev), torch.randint(0, classes, (b,), device=dev)

    return make


def _image_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    x, y = batch
    out = model(x)
    if hasattr(out, "logits"):
        out = out.logits
    if isinstance(out, (tuple, list)):
        out = out[0]
    return F.cross_entropy(out, y)


def _seg_batch(b: int, r: int, dev: torch.device):
    return torch.randn(b, 3, r, r, device=dev), torch.randint(0, 21, (b, r, r), device=dev)


def _seg_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    x, y = batch
    out = model(x)
    if isinstance(out, dict):
        out = out["out"]
    return F.cross_entropy(out, y)


def _video_batch(b: int, t: int, dev: torch.device):
    return torch.randn(b, 3, t, 112, 112, device=dev), torch.randint(0, 400, (b,), device=dev)


def _hf_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    return model(**batch).loss


def _text_cls_batch(vocab: int):
    def make(b: int, n: int, dev: torch.device):
        return {
            "input_ids": torch.randint(0, vocab, (b, n), device=dev),
            "attention_mask": torch.ones(b, n, dtype=torch.long, device=dev),
            "labels": torch.randint(0, 2, (b,), device=dev),
        }

    return make


def _text_lm_batch(vocab: int):
    def make(b: int, n: int, dev: torch.device):
        ids = torch.randint(0, vocab, (b, n), device=dev)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    return make


def _seq2seq_batch(vocab: int):
    def make(b: int, n: int, dev: torch.device):
        return {
            "input_ids": torch.randint(2, vocab, (b, n), device=dev),
            "attention_mask": torch.ones(b, n, dtype=torch.long, device=dev),
            "labels": torch.randint(2, vocab, (b, n), device=dev),
        }

    return make


def _audio_batch(b: int, n: int, dev: torch.device):
    return {
        "input_values": torch.randn(b, n, device=dev),
        "labels": torch.randint(0, 2, (b,), device=dev),
    }


# ----------------------------------------------------------------------------
# custom models
# ----------------------------------------------------------------------------


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab: int = 32000, dim: int = 512, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.rnn = nn.LSTM(dim, dim, layers, batch_first=True)
        self.out = nn.Linear(dim, vocab)

    def forward(self, ids):
        h, _ = self.rnn(self.emb(ids))
        return self.out(h)


class GRUClassifier(nn.Module):
    def __init__(self, vocab: int = 32000, dim: int = 384):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, 2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * dim, 2)

    def forward(self, ids):
        h, _ = self.rnn(self.emb(ids))
        return self.out(h.mean(1))


class TorchEncoderLM(nn.Module):
    def __init__(self, vocab: int = 32000, dim: int = 512, layers: int = 6, heads: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(4096, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(dim, vocab)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        return self.out(self.enc(self.emb(ids) + self.pos(pos)))


class TorchSeq2Seq(nn.Module):
    def __init__(self, vocab: int = 32000, dim: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.core = nn.Transformer(dim, 8, 4, 4, 2048, batch_first=True)
        self.out = nn.Linear(dim, vocab)

    def forward(self, ids):
        e = self.emb(ids)
        return self.out(self.core(e, e))


def _lm_plain_batch(vocab: int = 32000):
    def make(b: int, n: int, dev: torch.device):
        ids = torch.randint(0, vocab, (b, n), device=dev)
        return ids, ids

    return make


def _lm_plain_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    ids, y = batch
    out = model(ids)
    return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))


def _cls_plain_batch(b: int, n: int, dev: torch.device):
    return torch.randint(0, 32000, (b, n), device=dev), torch.randint(0, 2, (b,), device=dev)


def _cls_plain_loss(model: nn.Module, batch: Any) -> torch.Tensor:
    x, y = batch
    return F.cross_entropy(model(x), y)


class _Double(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )


class UNet(nn.Module):
    def __init__(self, width: int = 32, classes: int = 21):
        super().__init__()
        w = width
        self.d1, self.d2, self.d3, self.d4 = (
            _Double(3, w),
            _Double(w, 2 * w),
            _Double(2 * w, 4 * w),
            _Double(4 * w, 8 * w),
        )
        self.u3, self.u2, self.u1 = (
            _Double(12 * w, 4 * w),
            _Double(6 * w, 2 * w),
            _Double(3 * w, w),
        )
        self.head = nn.Conv2d(w, classes, 1)

    def forward(self, x):
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        d = self.d4(F.max_pool2d(c, 2))
        c = self.u3(torch.cat([F.interpolate(d, scale_factor=2.0), c], 1))
        b = self.u2(torch.cat([F.interpolate(c, scale_factor=2.0), b], 1))
        a = self.u1(torch.cat([F.interpolate(b, scale_factor=2.0), a], 1))
        return self.head(a)


class Conv1dNet(nn.Module):
    def __init__(self, width: int = 128, depth: int = 10):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(1, width, 7, stride=2, padding=3)]
        for i in range(depth):
            layers += [
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Conv1d(width, width, 3, padding=2 ** (i % 5), dilation=2 ** (i % 5)),
            ]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.body(x.unsqueeze(1)).mean(-1))


def _wave_batch(b: int, n: int, dev: torch.device):
    return torch.randn(b, n, device=dev), torch.randint(0, 2, (b,), device=dev)


class MLP(nn.Module):
    def __init__(self, width: int = 4096, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.LazyLinear(width), nn.ReLU(), nn.Dropout(0.1)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, 10)

    def forward(self, x):
        return self.head(self.body(x))


def _tab_batch(b: int, n: int, dev: torch.device):
    return torch.randn(b, n, device=dev), torch.randint(0, 10, (b,), device=dev)


# ----------------------------------------------------------------------------
# zoo
# ----------------------------------------------------------------------------

IMG_GRID = (128, 160, 192, 256, 320, 384)
TXT_GRID = (32, 64, 256, 512, 1024)


def _tv(name: str, family: str, flexible: bool = True, default: int = 224) -> Spec:
    def build():
        import torchvision

        return getattr(torchvision.models, name)(weights=None)

    grid = tuple(g for g in IMG_GRID if g >= min(default, 128)) if flexible else ()
    return Spec(name, family, "image", build, _image_batch(), _image_loss, "resolution", default, grid)


def _tvseg(name: str) -> Spec:
    def build():
        import torchvision

        return getattr(torchvision.models.segmentation, name)(weights=None, weights_backbone=None)

    return Spec(name, "segmentation", "segmentation", build, _seg_batch, _seg_loss, "resolution", 224, IMG_GRID)


def _tvvideo(name: str) -> Spec:
    def build():
        import torchvision

        return getattr(torchvision.models.video, name)(weights=None)

    return Spec(name, "video", "video", build, _video_batch, _cls_plain_loss, "frames", 8, (4, 16, 32))


def _timm(name: str, family: str) -> Spec:
    def build():
        import timm

        return timm.create_model(name, pretrained=False)

    def make(b: int, r: int, dev: torch.device):
        return torch.randn(b, 3, r, r, device=dev), torch.randint(0, 1000, (b,), device=dev)

    res = 224
    for token in name.split("_"):
        if token.isdigit() and 200 <= int(token) <= 400:
            res = int(token)
    return Spec("timm_" + name, family, "image", build, make, _image_loss, "resolution", res, ())


def _hf(name, family, domain, cls, cfg, batch, default=128, grid=TXT_GRID, axis="length", **kw) -> Spec:
    def build():
        import transformers

        config = getattr(transformers, cfg)(**kw)
        return getattr(transformers, cls)(config)

    return Spec("hf_" + name, family, domain, build, batch, _hf_loss, axis, default, grid)


def _enc(name, cfg, vocab=30522, family="text-encoder", **kw) -> Spec:
    cls = cfg.replace("Config", "ForSequenceClassification")
    kw.setdefault("max_position_embeddings", 2048)
    return _hf(name, family, "text-encoder", cls, cfg, _text_cls_batch(vocab), vocab_size=vocab, **kw)


def _dec(name, cfg, cls, vocab=32000, **kw) -> Spec:
    return _hf(name, "text-decoder", "text-decoder", cls, cfg, _text_lm_batch(vocab), vocab_size=vocab, **kw)


def _s2s(name, cfg, cls, vocab=32000, **kw) -> Spec:
    return _hf(name, "seq2seq", "seq2seq", cls, cfg, _seq2seq_batch(vocab), vocab_size=vocab, **kw)


def _whisper(name: str, **kw) -> Spec:
    def make(b: int, n: int, dev: torch.device):
        return {
            "input_features": torch.randn(b, 80, 3000, device=dev),
            "labels": torch.randint(0, 50000, (b, n), device=dev),
        }

    return _hf(name, "audio", "audio", "WhisperForConditionalGeneration", "WhisperConfig", make, 64, (), **kw)


def _clip() -> Spec:
    def build():
        import transformers

        return transformers.CLIPModel(transformers.CLIPConfig())

    def make(b: int, r: int, dev: torch.device):
        return {
            "input_ids": torch.randint(0, 49000, (b, 77), device=dev),
            "pixel_values": torch.randn(b, 3, r, r, device=dev),
            "return_loss": True,
        }

    return Spec("hf_clip_b32", "multimodal", "multimodal", build, make, _hf_loss, "resolution", 224, ())


def _segformer() -> Spec:
    def build():
        import transformers

        return transformers.SegformerForSemanticSegmentation(transformers.SegformerConfig(num_labels=21))

    def make(b: int, r: int, dev: torch.device):
        return {
            "pixel_values": torch.randn(b, 3, r, r, device=dev),
            "labels": torch.randint(0, 21, (b, r, r), device=dev),
        }

    return Spec("hf_segformer_b0", "segmentation", "segmentation", build, make, _hf_loss, "resolution", 224, IMG_GRID)


def _ast() -> Spec:
    def build():
        import transformers

        return transformers.ASTForAudioClassification(transformers.ASTConfig())

    def make(b: int, n: int, dev: torch.device):
        return {
            "input_values": torch.randn(b, 1024, 128, device=dev),
            "labels": torch.randint(0, 2, (b,), device=dev),
        }

    return Spec("hf_ast", "audio", "audio", build, make, _hf_loss, "length", 1024, ())


def _custom(name, family, domain, build, batch, loss, axis, default, grid) -> Spec:
    return Spec("custom_" + name, family, domain, build, batch, loss, axis, default, grid)


def all_specs() -> list[Spec]:
    s: list[Spec] = []
    # torchvision classification
    s += [_tv("alexnet", "plain-cnn"), _tv("vgg11", "plain-cnn"), _tv("vgg16", "plain-cnn")]
    s += [_tv("vgg19_bn", "plain-cnn"), _tv("squeezenet1_1", "plain-cnn")]
    s += [_tv(n, "resnet") for n in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")]
    s += [_tv(n, "resnet") for n in ("resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2")]
    s += [_tv(n, "densenet") for n in ("densenet121", "densenet169", "densenet201")]
    s += [_tv(n, "mobile-cnn") for n in ("mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large")]
    s += [_tv(n, "mobile-cnn") for n in ("shufflenet_v2_x1_0", "mnasnet1_0")]
    s += [_tv(n, "efficientnet") for n in ("efficientnet_b0", "efficientnet_b2", "efficientnet_b4")]
    s += [_tv(n, "efficientnet") for n in ("efficientnet_v2_s", "efficientnet_v2_m")]
    s += [_tv(n, "regnet") for n in ("regnet_y_400mf", "regnet_y_1_6gf", "regnet_y_8gf", "regnet_x_3_2gf")]
    s += [_tv(n, "convnext") for n in ("convnext_tiny", "convnext_small", "convnext_base")]
    s += [_tv("googlenet", "inception"), _tv("inception_v3", "inception", default=299)]
    s += [_tv(n, "vit", flexible=False) for n in ("vit_b_16", "vit_b_32", "vit_l_16")]
    s += [_tv(n, "swin", flexible=False) for n in ("swin_t", "swin_s", "swin_v2_t", "maxvit_t")]
    s += [_tvseg(n) for n in ("fcn_resnet50", "deeplabv3_resnet50")]
    s += [_tvseg(n) for n in ("deeplabv3_mobilenet_v3_large", "lraspp_mobilenet_v3_large")]
    s += [_tvvideo(n) for n in ("r3d_18", "mc3_18", "r2plus1d_18")]
    # timm
    timm_models = {
        "vit": ("deit_tiny_patch16_224", "deit_small_patch16_224", "vit_small_patch16_224",
                "beit_base_patch16_224", "cait_s24_224", "xcit_small_12_p16_224",
                "tnt_s_patch16_224", "eva02_tiny_patch14_224", "volo_d1_224"),
        "hier-vit": ("coat_lite_small", "davit_tiny", "pvt_v2_b2", "twins_svt_small", "pit_s_224",
                     "nest_tiny", "gcvit_tiny", "levit_256", "focalnet_tiny_srf",
                     "efficientformer_l1", "mobilevit_s", "mobilevitv2_100", "edgenext_small",
                     "efficientvit_b1", "fastvit_t8"),
        "mlp-like": ("mixer_b16_224", "resmlp_24_224", "convmixer_768_32", "poolformer_s24",
                     "sequencer2d_s"),
        "timm-cnn": ("ghostnet_100", "hrnet_w18", "inception_resnet_v2", "dm_nfnet_f0",
                     "repvgg_b1", "res2net50_26w_4s", "resnest50d", "rexnet_100", "dla60",
                     "dpn68", "cspdarknet53", "darknet53", "xception41", "tresnet_m",
                     "lcnet_100", "hgnetv2_b2", "convnextv2_tiny", "mambaout_tiny"),
    }
    for family, names in timm_models.items():
        s += [_timm(n, family) for n in names]
    # text encoders
    s += [
        _enc("bert_tiny", "BertConfig", hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512),
        _enc("bert_mini", "BertConfig", hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024),
        _enc("bert_small", "BertConfig", hidden_size=512, num_hidden_layers=4, num_attention_heads=8, intermediate_size=2048),
        _enc("bert_medium", "BertConfig", hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048),
        _enc("bert_base", "BertConfig"),
        _enc("bert_large", "BertConfig", hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
        _enc("roberta_base", "RobertaConfig", vocab=50265),
        _enc("distilbert", "DistilBertConfig"),
        _enc("albert_base", "AlbertConfig", vocab=30000, hidden_size=768, num_attention_heads=12, intermediate_size=3072),
        _enc("electra_small", "ElectraConfig"),
        _enc("electra_base", "ElectraConfig", embedding_size=768, hidden_size=768, num_attention_heads=12, intermediate_size=3072),
        _enc("deberta_v2_base", "DebertaV2Config", vocab=128100, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
        _enc("mpnet_base", "MPNetConfig", vocab=30527),
        _enc("convbert_base", "ConvBertConfig"),
        _enc("mobilebert", "MobileBertConfig"),
        _enc("squeezebert", "SqueezeBertConfig"),
        _enc("xlnet_base", "XLNetConfig", vocab=32000),
        _enc("funnel", "FunnelConfig"),
    ]
    # decoders
    s += [
        _dec("gpt2_distil", "GPT2Config", "GPT2LMHeadModel", 50257, n_layer=6, n_positions=2048),
        _dec("gpt2_small", "GPT2Config", "GPT2LMHeadModel", 50257, n_positions=2048),
        _dec("gpt2_medium", "GPT2Config", "GPT2LMHeadModel", 50257, n_positions=2048, n_embd=1024, n_layer=24, n_head=16),
        _dec("gpt2_large", "GPT2Config", "GPT2LMHeadModel", 50257, n_positions=2048, n_embd=1280, n_layer=36, n_head=20),
        _dec("opt_125m", "OPTConfig", "OPTForCausalLM", 50272, max_position_embeddings=2048),
        _dec("pythia_70m", "GPTNeoXConfig", "GPTNeoXForCausalLM", 50304, hidden_size=512, num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048),
        _dec("pythia_160m", "GPTNeoXConfig", "GPTNeoXForCausalLM", 50304, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
        _dec("pythia_410m", "GPTNeoXConfig", "GPTNeoXForCausalLM", 50304, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
        _dec("llama_s", "LlamaConfig", "LlamaForCausalLM", hidden_size=512, intermediate_size=1376, num_hidden_layers=8, num_attention_heads=8),
        _dec("llama_m", "LlamaConfig", "LlamaForCausalLM", hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12),
        _dec("llama_l", "LlamaConfig", "LlamaForCausalLM", hidden_size=1024, intermediate_size=2752, num_hidden_layers=16, num_attention_heads=16),
        _dec("llama_1b", "LlamaConfig", "LlamaForCausalLM", hidden_size=2048, intermediate_size=5632, num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4),
        _dec("mistral_s", "MistralConfig", "MistralForCausalLM", hidden_size=1024, intermediate_size=3584, num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=4),
        _dec("qwen2_05b", "Qwen2Config", "Qwen2ForCausalLM", 151936, hidden_size=896, intermediate_size=4864, num_hidden_layers=24, num_attention_heads=14, num_key_value_heads=2),
        _dec("bloom_560m", "BloomConfig", "BloomForCausalLM", 250880, hidden_size=1024, n_layer=24, n_head=16),
        _dec("gptneo_125m", "GPTNeoConfig", "GPTNeoForCausalLM", 50257, hidden_size=768, num_layers=12, num_heads=12, attention_types=[[["global", "local"], 6]]),
    ]
    # sequence to sequence
    s += [
        _s2s("t5_small", "T5Config", "T5ForConditionalGeneration", 32128, decoder_start_token_id=0),
        _s2s("t5_base", "T5Config", "T5ForConditionalGeneration", 32128, decoder_start_token_id=0, d_model=768, d_ff=3072, num_layers=12, num_heads=12),
        _s2s("t5_large", "T5Config", "T5ForConditionalGeneration", 32128, decoder_start_token_id=0, d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
        _s2s("bart_base", "BartConfig", "BartForConditionalGeneration", 50265, d_model=768, encoder_layers=6, decoder_layers=6, encoder_attention_heads=12, decoder_attention_heads=12, encoder_ffn_dim=3072, decoder_ffn_dim=3072, max_position_embeddings=2048),
        _s2s("bart_large", "BartConfig", "BartForConditionalGeneration", 50265, max_position_embeddings=2048),
        _s2s("marian", "MarianConfig", "MarianMTModel", 58101, max_position_embeddings=2048),
        _s2s("pegasus", "PegasusConfig", "PegasusForConditionalGeneration", 50265, max_position_embeddings=2048),
    ]
    # audio and others
    aud = dict(domain="audio", batch=_audio_batch, default=32000, grid=(16000, 64000, 128000), axis="samples")
    s += [
        _hf("wav2vec2_base", "audio", cls="Wav2Vec2ForSequenceClassification", cfg="Wav2Vec2Config", **aud),
        _hf("hubert_base", "audio", cls="HubertForSequenceClassification", cfg="HubertConfig", **aud),
        _hf("wavlm_base", "audio", cls="WavLMForSequenceClassification", cfg="WavLMConfig", **aud),
        _whisper("whisper_tiny"),
        _whisper("whisper_base", d_model=512, encoder_layers=6, decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=2048, decoder_ffn_dim=2048),
        _whisper("whisper_small", d_model=768, encoder_layers=12, decoder_layers=12, encoder_attention_heads=12, decoder_attention_heads=12, encoder_ffn_dim=3072, decoder_ffn_dim=3072),
        _ast(),
        _clip(),
        _segformer(),
    ]
    # custom
    lm = (_lm_plain_batch(), _lm_plain_loss, "length", 128, TXT_GRID)
    s += [
        _custom("lstm_lm", "recurrent", "recurrent", LSTMLanguageModel, *lm),
        _custom("lstm_lm_wide", "recurrent", "recurrent", lambda: LSTMLanguageModel(dim=1024, layers=3), *lm),
        _custom("gru_cls", "recurrent", "recurrent", GRUClassifier, _cls_plain_batch, _cls_plain_loss, "length", 128, TXT_GRID),
        _custom("torch_encoder_lm", "torch-transformer", "text-decoder", TorchEncoderLM, *lm),
        _custom("torch_encoder_lm_wide", "torch-transformer", "text-decoder", lambda: TorchEncoderLM(dim=768, layers=12, heads=12), *lm),
        _custom("torch_seq2seq", "torch-transformer", "seq2seq", TorchSeq2Seq, *lm),
        _custom("unet_32", "unet", "segmentation", UNet, _seg_batch, _seg_loss, "resolution", 224, IMG_GRID),
        _custom("unet_64", "unet", "segmentation", lambda: UNet(64), _seg_batch, _seg_loss, "resolution", 224, IMG_GRID),
        _custom("conv1d", "conv1d", "audio", Conv1dNet, _wave_batch, _cls_plain_loss, "samples", 32000, (16000, 64000, 128000)),
        _custom("mlp", "mlp", "tabular", MLP, _tab_batch, _cls_plain_loss, "features", 512, (128, 2048)),
    ]
    return s


def get_spec(name: str) -> Spec:
    for spec in all_specs():
        if spec.name == name:
            return spec
    raise KeyError(name)


if __name__ == "__main__":
    for spec in all_specs():
        print(spec.name, spec.family, spec.domain, spec.axis, spec.default)
