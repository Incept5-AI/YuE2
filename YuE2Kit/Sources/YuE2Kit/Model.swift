import Foundation
import MLX
import MLXNN

// Port of yue2_model.py: a Qwen3-style decoder whose every layer carries two attention/MLP paths
// (AR for text+codec tokens, NAR for the acoustic flow-matching state). Weight keys match the checkpoint.

public struct ModelConfig: Codable, Sendable {
    public struct Quant: Codable, Sendable {
        public var group_size: Int
        public var bits: Int
        public var nar_bits: Int?
    }
    public var hidden_size: Int
    public var num_hidden_layers: Int
    public var num_attention_heads: Int
    public var num_key_value_heads: Int
    public var head_dim: Int
    public var intermediate_size: Int
    public var vocab_size: Int
    public var rms_norm_eps: Float
    public var rope_theta: Float
    public var latent_dim: Int
    public var max_latent_frames: Int
    public var timestep_shift: Float
    public var quantization: Quant?
}

/// mlx-python's nn.silu is uncompiled `x * sigmoid(x)` (two bf16 roundings); mlx-swift's `silu` is a fused kernel
/// that rounds once. Match Python so logits agree with the reference runner.
@inline(__always) func pySilu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// Append-only per-layer cache, grown in 256-slot steps (mlx_lm pattern).
final class KVCache {
    static let step = 256
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0
    /// Slots to allocate on first use. Growing by concatenation briefly holds old + new buffers (about 2x the
    /// cache, ~2.4 GB for a full song); on a phone with a ~6 GB budget that spike is what gets the app killed.
    let capacity: Int
    init(capacity: Int = 0) { self.capacity = capacity }

    func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset
        let (B, H, T, D) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3))
        if keys == nil || prev + T > keys!.dim(2) {
            let want = keys == nil ? max(T, capacity) : T
            let n = ((want + Self.step - 1) / Self.step) * Self.step
            let nk = MLXArray.zeros([B, H, n, D], dtype: k.dtype), nv = MLXArray.zeros([B, H, n, D], dtype: v.dtype)
            if var ck = keys, var cv = values {
                if prev % Self.step != 0 { ck = ck[.ellipsis, ..<prev, 0...]; cv = cv[.ellipsis, ..<prev, 0...] }
                keys = concatenated([ck, nk], axis: 2); values = concatenated([cv, nv], axis: 2)
            } else {
                keys = nk; values = nv
            }
        }
        offset += T
        keys![.ellipsis, prev..<offset, 0...] = k
        values![.ellipsis, prev..<offset, 0...] = v
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }
}

final class Attention: Module {
    let nHeads: Int, nKV: Int, headDim: Int, ropeBase: Float, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: ModelConfig) {
        nHeads = c.num_attention_heads; nKV = c.num_key_value_heads; headDim = c.head_dim
        ropeBase = c.rope_theta; scale = pow(Float(headDim), -0.5)
        let h = c.hidden_size, d = headDim
        _qProj.wrappedValue = Linear(h, nHeads * d, bias: false)
        _kProj.wrappedValue = Linear(h, nKV * d, bias: false)
        _vProj.wrappedValue = Linear(h, nKV * d, bias: false)
        _oProj.wrappedValue = Linear(nHeads * d, h, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: d, eps: c.rms_norm_eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: d, eps: c.rms_norm_eps)
    }

    func projectQKV(_ x: MLXArray, offset: Int) -> (MLXArray, MLXArray, MLXArray) {
        let (B, T) = (x.dim(0), x.dim(1))
        let q = qNorm(qProj(x).reshaped(B, T, nHeads, -1)).transposed(0, 2, 1, 3)
        let k = kNorm(kProj(x).reshaped(B, T, nKV, -1)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, T, nKV, -1).transposed(0, 2, 1, 3)
        func rope(_ a: MLXArray) -> MLXArray {
            MLXFast.RoPE(a, dimensions: headDim, traditional: false, base: ropeBase, scale: 1.0, offset: offset)
        }
        return (rope(q), rope(k), v)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, causal: Bool) -> MLXArray {
        let (B, T) = (q.dim(0), q.dim(2))
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: causal ? .causal : .none)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(B, T, -1))
    }
}

final class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: ModelConfig) {
        _gate.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _up.wrappedValue = Linear(c.hidden_size, c.intermediate_size, bias: false)
        _down.wrappedValue = Linear(c.intermediate_size, c.hidden_size, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(pySilu(gate(x)) * up(x)) }
}

final class DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attn: Attention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "nar_input_layernorm") var narInputNorm: RMSNorm
    @ModuleInfo(key: "nar_self_attn") var narAttn: Attention
    @ModuleInfo(key: "nar_pre_mlp_layernorm") var narPreMLPNorm: RMSNorm
    @ModuleInfo(key: "nar_mlp") var narMLP: MLP

    init(_ c: ModelConfig) {
        let h = c.hidden_size, eps = c.rms_norm_eps
        _inputNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        _attn.wrappedValue = Attention(c)
        _postNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        _mlp.wrappedValue = MLP(c)
        _narInputNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        _narAttn.wrappedValue = Attention(c)
        _narPreMLPNorm.wrappedValue = RMSNorm(dimensions: h, eps: eps)
        _narMLP.wrappedValue = MLP(c)
    }

    func ar(_ x: MLXArray, cache: KVCache, causal: Bool) -> MLXArray {
        let (q, k0, v0) = attn.projectQKV(inputNorm(x), offset: cache.offset)
        let (k, v) = cache.update(k0, v0)
        let h = x + attn.attend(q, k, v, causal: causal)
        return h + mlp(postNorm(h))
    }

    func arPrefillKV(_ x: MLXArray) -> (MLXArray, (MLXArray, MLXArray)) {
        let (q, k, v) = attn.projectQKV(inputNorm(x), offset: 0)
        let h = x + attn.attend(q, k, v, causal: true)
        return (h + mlp(postNorm(h)), (k, v))
    }

    func nar(_ x: MLXArray, arK: MLXArray, arV: MLXArray, offset: Int) -> MLXArray {
        let (q, k0, v0) = narAttn.projectQKV(narInputNorm(x), offset: offset)
        let k = concatenated([arK, k0], axis: 2), v = concatenated([arV, v0], axis: 2)
        let h = x + narAttn.attend(q, k, v, causal: false)
        return h + narMLP(narPreMLPNorm(h))
    }
}

final class Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: ModelConfig) {
        _embed.wrappedValue = Embedding(embeddingCount: c.vocab_size, dimensions: c.hidden_size)
        _layers.wrappedValue = (0..<c.num_hidden_layers).map { _ in DecoderLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden_size, eps: c.rms_norm_eps)
    }
}

final class TimestepEmbedder: Module {
    let freqSize = 256
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hidden: Int) {
        _fc1.wrappedValue = Linear(256, hidden)
        _fc2.wrappedValue = Linear(hidden, hidden)
    }

    func callAsFunction(_ t: MLXArray, dtype: DType) -> MLXArray {
        let half = freqSize / 2
        let freqs = exp(-Float(log(10000.0)) * MLXArray(0..<half).asType(.float32) / Float(half))
        let args = t.asType(.float32).reshaped(-1, 1) * freqs[.newAxis]
        let emb = concatenated([cos(args), sin(args)], axis: -1).asType(dtype)
        return fc2(pySilu(fc1(emb)))
    }
}

final class LatentPosEmbed: Module {
    @ParameterInfo(key: "pe") var pe: MLXArray
    init(maxFrames: Int, hidden: Int) { _pe.wrappedValue = MLXArray.zeros([maxFrames, hidden]) }
}

public final class Yue2Model: Module {
    let config: ModelConfig
    @ModuleInfo(key: "model") var model: Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "llm2vae") var llm2vae: Linear
    @ModuleInfo(key: "vae2llm") var vae2llm: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: TimestepEmbedder
    @ModuleInfo(key: "latent_pos_embed") var latentPosEmbed: LatentPosEmbed

    init(_ c: ModelConfig) {
        config = c
        _model.wrappedValue = Backbone(c)
        _lmHead.wrappedValue = Linear(c.hidden_size, c.vocab_size, bias: false)
        _llm2vae.wrappedValue = Linear(c.hidden_size, c.latent_dim)
        _vae2llm.wrappedValue = Linear(c.latent_dim, c.hidden_size)
        _timeEmbedder.wrappedValue = TimestepEmbedder(hidden: c.hidden_size)
        _latentPosEmbed.wrappedValue = LatentPosEmbed(maxFrames: c.max_latent_frames, hidden: c.hidden_size)
    }

    var layerCount: Int { model.layers.count }

    /// tokens [B,T] → logits [B,V] of the last position.
    func arStep(_ tokens: MLXArray, caches: [KVCache]) -> MLXArray {
        var x = model.embed(tokens)
        let causal = tokens.dim(1) > 1
        for (layer, cache) in zip(model.layers, caches) { x = layer.ar(x, cache: cache, causal: causal) }
        x = model.norm(x[0..., (-1)..., 0...])
        return lmHead(x)[0..., -1, 0...]
    }

    /// Run the AR path once over the chunk prefix; return per-layer (k, v).
    func narPrefill(_ arTokens: [Int]) -> [(MLXArray, MLXArray)] {
        var x = model.embed(MLXArray(arTokens.map(Int32.init)).reshaped(1, -1))
        var cache: [(MLXArray, MLXArray)] = []
        for layer in model.layers {
            let (h, kv) = layer.arPrefillKV(x)
            x = h
            cache.append(kv)
        }
        eval(cache.flatMap { [$0.0, $0.1] })
        return cache
    }

    /// Flow-matching velocity for `state` [T,64] at logit-time `rawT`.
    func narVelocity(_ state: MLXArray, rawT: Double, arCache: [(MLXArray, MLXArray)], arLength: Int) -> MLXArray {
        let shift = config.timestep_shift
        let sig = sigmoid(MLXArray(Float(rawT)).asType(state.dtype))
        let t = shift * sig / (1 + (shift - 1) * sig)
        let n = state.dim(0) + 2
        var x = vae2llm(padded(state, widths: [IntOrPair((1, 1)), IntOrPair((0, 0))])[.newAxis])
        x = x + timeEmbedder(t, dtype: state.dtype)[.newAxis]
        let pos = minimum(MLXArray(0..<n), MLXArray(Int32(config.max_latent_frames - 1)))
        x = x + latentPosEmbed.pe[pos][.newAxis]
        for (layer, (k, v)) in zip(model.layers, arCache) { x = layer.nar(x, arK: k, arV: v, offset: arLength) }
        return llm2vae(model.norm(x))[0, 1..<(n - 1), 0...]
    }
}

/// `quantizeEmbedding`: also 4-bit-quantize the 756 MB bf16 token-embedding table (the checkpoint only quantizes
/// Linear layers). Saves ~550 MB for memory-tight phones; lm_head is already 4-bit, so the effect is small.
public func loadModel(directory: URL, quantizeEmbedding: Bool = false) throws -> Yue2Model {
    let cfg = try JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: directory.appending(path: "config.json")))
    let model = Yue2Model(cfg)
    let weights = try loadArrays(url: directory.appending(path: "model.safetensors"))
    if let q = cfg.quantization {
        let narBits = q.nar_bits ?? q.bits
        quantize(model: model, filter: { path, module in
            guard module is Linear, weights["\(path).scales"] != nil else { return nil }
            return (q.group_size, path.contains(".nar_") ? narBits : q.bits, .affine)
        })
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    if quantizeEmbedding {
        quantize(model: model, filter: { path, module in
            path == "model.embed_tokens" && module is Embedding && !(module is QuantizedEmbedding) ? (64, 4, .affine) : nil
        })
    }
    eval(model.parameters())
    return model
}
