import Foundation
import MLX
import MLXNN

// Port of generate.py: symbolic ABC planning (cot=melody/full), semantic codec AR decode with CFG,
// midpoint flow matching, tiled VAE decode.

public let EOD = 151643
public let ABC_START = 151847, ABC_END = 151848
public let MUSIC_START = 151851, MUSIC_END = 151852
public let CODEC_OFFSET = 151853, CODEC_SIZE = 32768
public let VOCAB = 184704, CONTEXT = 24576
public let sampleRate = 48000
/// One semantic token is 40 ms of audio (1920 samples at 48 kHz).
public let secondsPerToken = Double(frameSamples) / Double(sampleRate)

public enum Planning: String, CaseIterable, Codable, Sendable {
    case off, melody, full
    var instruction: String {
        switch self {
        case .off: "Generate music with codec tokens from the given conditions."
        case .melody: "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
        case .full: "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
        }
    }
}

public struct Sampling: Codable, Sendable, Equatable {
    public var temperature: Float = 1.0
    public var top_p: Float = 0.95
    public var top_k: Int = 100
    public var repetition_penalty: Float = 1.2
    public var penalty_window: Int = 50
    public var min_tokens: Int = 200
    public var max_tokens: Int = 9000

    public init(temperature: Float = 1.0, top_p: Float = 0.95, top_k: Int = 100, repetition_penalty: Float = 1.2,
                penalty_window: Int = 50, min_tokens: Int = 200, max_tokens: Int = 9000) {
        self.temperature = temperature; self.top_p = top_p; self.top_k = top_k
        self.repetition_penalty = repetition_penalty; self.penalty_window = penalty_window
        self.min_tokens = min_tokens; self.max_tokens = max_tokens
    }

    public static let abcDefault = Sampling(temperature: 0.7, top_p: 0.9, top_k: 30, repetition_penalty: 1.005,
                                            penalty_window: 100, min_tokens: 32, max_tokens: 4096)
    public static let semanticDefault = Sampling()
}

public enum Stage: String, Sendable { case loading, planning, semantic, acoustic, decoding, done }

public struct Progress: Sendable {
    public var stage: Stage
    public var fraction: Double      // within the stage
    public var detail: String
    public init(stage: Stage, fraction: Double, detail: String) { self.stage = stage; self.fraction = fraction; self.detail = detail }
}

public struct CancelledError: Error, LocalizedError {
    public var errorDescription: String? { "Cancelled" }
}

public struct RenderRequest: Codable, Sendable {
    public var style: String
    public var lyrics: String
    public var planning: Planning = .full
    public var seed: Int = 831001
    public var abc: String? = nil
    public var cfgScale: Float? = nil          // nil = auto (1.0 when planning, 1.01 when off)
    public var steps: Int = 32
    public var abcSampling: Sampling = .abcDefault
    public var semanticSampling: Sampling = .semanticDefault

    public init(style: String, lyrics: String) { self.style = style; self.lyrics = lyrics }
}

public struct RenderResult: Sendable {
    public var audio: [Float]          // interleaved stereo, 48 kHz
    public var frames: Int
    public var codec: [Int]
    public var abc: String?
    public var abcTruncated: Bool
    public var semanticTruncated: Bool
    public var timings: [String: Double]
    public var peakMemory: Int
    public var stagePeaksGB: [String: Double] = [:]
    public var silencedGroups = 0          // score groups whose vocal line was taken out (instrumental)
}

func tokenPrefix(_ tok: Tokenizer, style: String, lyrics: String, planning: Planning, abcIDs: [Int]?) -> [Int] {
    let base = [EOD] + tok.encode("\(planning.instruction)\n[Tags]\n\(style)\n[Lyrics]\n\(lyrics)\n")
    if planning == .off { return base + [ABC_START, ABC_END, MUSIC_START] }
    guard let abcIDs else { return base + [ABC_START] }
    return base + [ABC_START] + abcIDs + [ABC_END, MUSIC_START]
}

func negativePrefix(_ tok: Tokenizer, planning: Planning, abcIDs: [Int]) -> [Int] {
    let base = [EOD] + tok.encode(planning.instruction)
    if planning == .off { return base + [MUSIC_START] }
    return base + [ABC_START] + abcIDs + [ABC_END, MUSIC_START]
}

enum Phase { case abc, semantic }

func allowedMask(_ phase: Phase, dtype: DType) -> MLXArray {
    let idx = MLXArray.arange(VOCAB, dtype: .int32)
    let end = phase == .abc ? ABC_END : MUSIC_END
    let ok = phase == .abc ? (idx .< Int32(EOD)) : ((idx .>= Int32(CODEC_OFFSET)) .&& (idx .< Int32(CODEC_OFFSET + CODEC_SIZE)))
    return which(ok .|| (idx .== Int32(end)), MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
}

/// Positions the top-p filter may remove (always keep the best 1, or 3 on the legacy cot=off path). Built once per
/// generation: rebuilding a 184,704-element index on the CPU every token cost ~11 ms/token (65 vs 240 tok/s).
func topPRemovable(legacyOff: Bool) -> MLXArray {
    MLXArray.arange(VOCAB, dtype: .int32) .>= Int32(legacyOff ? 3 : 1)
}

func distribution(_ logits: MLXArray, _ s: Sampling, history: [Int], step: Int, phase: Phase, allowed: MLXArray,
                  legacyOff: Bool, removable: MLXArray) -> MLXArray {
    // Historical cot=off arithmetic stays in BF16; symbolic/CFG paths upcast.
    var scores = legacyOff ? logits : logits.asType(.float32)
    let end = phase == .abc ? ABC_END : MUSIC_END
    scores = scores + allowed
    if step < s.min_tokens { scores[end] = MLXArray(-Float.infinity).asType(scores.dtype) }
    let recent = Array(history.suffix(s.penalty_window))
    if s.repetition_penalty != 1.0 && !recent.isEmpty {
        let freq = MLXArray.zeros([VOCAB], dtype: scores.dtype).at[MLXArray(recent.map(Int32.init))].add(1)
        let alpha = pow(MLXArray(s.repetition_penalty).asType(scores.dtype), freq)
        scores = which(scores .< 0, scores * alpha, scores / alpha)
    }
    if s.temperature == 0 { return scores }
    if s.temperature != 1 { scores = scores / s.temperature }
    let kth = VOCAB - min(s.top_k, VOCAB)
    let threshold = partitioned(scores, kth: kth)[kth]
    scores = which(scores .< threshold, -Float.infinity, scores)
    if s.top_p < 1 {
        let order = argSort(-scores)
        let values = scores[order]
        let probs = softmax(values.asType(.float32), axis: -1)
        var removed = (cumsum(probs) - probs) .> s.top_p
        removed = removed .&& removable
        scores = putAlong(scores, order, values: which(removed, -Float.infinity, values), axis: 0)
    }
    return scores
}

func chunkRanges(frames: Int, prefixTokens: Int) throws -> [(Int, Int)] {
    let size = min((CONTEXT - prefixTokens - 3) / 2, CONTEXT)
    guard frames >= 1, size >= 1 else { throw PipelineError.message("Empty codec or prefix leaves no acoustic context") }
    return stride(from: 0, to: frames, by: size).map { ($0, min($0 + size, frames)) }
}

func logit(_ t: Double) -> Double {
    if t > 0 && t < 1 { return max(-20, min(20, Foundation.log(t / (1 - t)))) }
    return t >= 1 ? 20 : -20
}

public enum PipelineError: Error, LocalizedError {
    case message(String)
    public var errorDescription: String? { if case .message(let m) = self { m } else { nil } }
}

public final class Yue2Pipeline: @unchecked Sendable {
    public let directory: URL
    public let tokenizer: Tokenizer
    public private(set) var model: Yue2Model?
    public private(set) var vae: OobleckDecoder
    public var vaeCore = 128
    /// Free the 3.6 GB language model before the VAE decode (and reload it for the next song) when memory is tight.
    public var unloadModelForDecode = false
    public let quantizeEmbedding: Bool

    /// `lowMemory`: for phones (~6 GB app budget). Quantizes the embedding table and frees the language model
    /// before the VAE decode, which otherwise needs ~3.4 GB on top of the ~3.6 GB of weights.
    public init(directory: URL, lowMemory: Bool = false) throws {
        self.directory = directory
        quantizeEmbedding = lowMemory
        unloadModelForDecode = lowMemory
        tokenizer = try Tokenizer(mergeFile: directory.appending(path: "qwen.tiktoken"))
        vae = try loadVAE(directory: directory)  // small; load first so a bad file fails before minutes of AR work
        model = try loadModel(directory: directory, quantizeEmbedding: lowMemory)
    }

    func ensureModel() throws -> Yue2Model {
        if let model { return model }
        let m = try loadModel(directory: directory, quantizeEmbedding: quantizeEmbedding)
        model = m
        return m
    }

    /// Return (ids, truncated). Two KV caches when CFG is active (cond / uncond).
    func generateTokens(_ model: Yue2Model, prefix: [Int], _ s: Sampling, seed: Int, phase: Phase, negative: [Int]? = nil,
                        cfgScale: Float = 1, legacyOff: Bool = false, onToken: (Int) throws -> Void) throws -> ([Int], Bool) {
        if prefix.count + s.max_tokens > CONTEXT || (negative.map { $0.count + s.max_tokens > CONTEXT } ?? false) {
            throw PipelineError.message("Prefix + generation budget exceeds the 24576 context")
        }
        let useCFG = cfgScale != 1
        if useCFG && negative == nil { throw PipelineError.message("CFG requires a negative prefix") }
        let n = model.layerCount
        let caches = (0..<(useCFG ? 2 : 1)).map { i in
            (0..<n).map { _ in KVCache(capacity: (i == 0 ? prefix.count : negative!.count) + s.max_tokens + 1) }
        }
        var cond = model.arStep(MLXArray(prefix.map(Int32.init)).reshaped(1, -1), caches: caches[0])
        var uncond = useCFG ? model.arStep(MLXArray(negative!.map(Int32.init)).reshaped(1, -1), caches: caches[1]) : nil
        let end = phase == .abc ? ABC_END : MUSIC_END
        let allowed = allowedMask(phase, dtype: legacyOff ? cond.dtype : .float32)
        let removable = topPRemovable(legacyOff: legacyOff)
        eval(allowed, removable)
        var key = MLXRandom.key(UInt64(seed))
        var history: [Int] = [], eos = false
        for step in 0..<s.max_tokens {
            let logits = useCFG ? uncond! + cfgScale * (cond - uncond!) : cond
            let scores = distribution(logits[0], s, history: history, step: step, phase: phase, allowed: allowed, legacyOff: legacyOff, removable: removable)
            let token: Int
            if s.temperature == 0 {
                token = argMax(scores).item(Int.self)
            } else {
                let (k, sub) = MLXRandom.split(key: key)
                key = k
                token = MLXRandom.categorical(scores.asType(.float32), key: sub).item(Int.self)
            }
            try onToken(token)
            if token == end { eos = true; break }
            history.append(token)
            if step + 1 < s.max_tokens {
                let nxt = MLXArray([Int32(token)]).reshaped(1, 1)
                cond = model.arStep(nxt, caches: caches[0])
                if useCFG { uncond = model.arStep(nxt, caches: caches[1]) }
            }
        }
        return (history, !eos)
    }

    /// Latents [frames,64] float32 via midpoint ODE from t=1 → 0, solved per context chunk.
    func synthesize(_ model: Yue2Model, prefix: [Int], codec: [Int], seed: Int, steps: Int,
                    onStep: (Int, Int) throws -> Void) throws -> MLXArray {
        let noise = MLXRandom.normal([codec.count, 64], key: MLXRandom.key(UInt64(seed)))
        let dt = 1.0 / Double(steps)
        var out: [MLXArray] = []
        let chunks = try chunkRanges(frames: codec.count, prefixTokens: prefix.count)
        for (ci, (a, b)) in chunks.enumerated() {
            let arTokens = prefix + codec[a..<b].map { $0 + CODEC_OFFSET } + [MUSIC_END]
            let arCache = model.narPrefill(arTokens)
            var state = noise[a..<b].asType(.bfloat16)
            for step in 0..<steps {
                let t = 1.0 - Double(step) * dt
                let v1 = model.narVelocity(state, rawT: logit(t), arCache: arCache, arLength: arTokens.count)
                let mid = state - v1 * Float(dt / 2)
                state = state - model.narVelocity(mid, rawT: logit(t - dt / 2), arCache: arCache, arLength: arTokens.count) * Float(dt)
                eval(state)
                try onStep(ci * steps + step + 1, chunks.count * steps)
            }
            out.append(state.asType(.float32))
        }
        return concatenated(out)
    }

    /// The ABC planning phase alone (the UI's "Plan" button).
    public func plan(style: String, lyrics: String, planning: Planning, seed: Int, sampling: Sampling = .abcDefault,
                     isCancelled: @escaping () -> Bool = { false }, progress: (Progress) -> Void = { _ in }) throws -> (String, Bool) {
        guard planning != .off else { throw PipelineError.message("Planning is off") }
        let model = try ensureModel()
        var count = 0
        let (ids, trunc) = try generateTokens(model, prefix: tokenPrefix(tokenizer, style: style, lyrics: lyrics, planning: planning, abcIDs: nil),
                                              sampling, seed: seed, phase: .abc) { _ in
            if isCancelled() { throw CancelledError() }
            count += 1
            if count % 16 == 0 { progress(Progress(stage: .planning, fraction: Double(count) / Double(sampling.max_tokens), detail: "\(count) score tokens")) }
        }
        let abc = tokenizer.decode(ids)
        return (isInstrumental(style) ? silenceVocal(abc).0 : abc, trunc)
    }

    public func render(_ r: RenderRequest, isCancelled: @escaping () -> Bool = { false },
                       progress: @escaping (Progress) -> Void = { _ in }) throws -> RenderResult {
        if r.abc != nil && r.planning == .off { throw PipelineError.message("A score needs planning set to melody or full") }
        Memory.peakMemory = 0
        var timings: [String: Double] = [:]
        var clock = Date()
        var peaks: [String: Double] = [:]
        var silenced = 0
        func lap(_ name: String) {
            timings[name] = Date().timeIntervalSince(clock); clock = Date()
            peaks[name] = Double(Memory.peakMemory) / 1e9; Memory.peakMemory = 0
        }
        // Stages 1-3 run inside a closure so the local `model` reference ends before the decode; otherwise
        // unloadModelForDecode frees nothing and the phone holds weights + VAE peak together (~6.2 GB).
        let (latents, codecOut, abcText, abcTrunc, semTrunc): (MLXArray, [Int], String?, Bool, Bool) = try {
            if model == nil { progress(Progress(stage: .loading, fraction: 0, detail: "Loading model")) }
            let model = try ensureModel()
            lap("load")

            // 1. symbolic plan
            var abcIDs: [Int] = [], abcText: String? = nil, abcTrunc = false
            silenced = 0
            if r.planning != .off {
                if let abc = r.abc, !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    abcIDs = tokenizer.encode(abc); abcText = abc
                } else {
                    let start = Date(); var count = 0
                    (abcIDs, abcTrunc) = try generateTokens(model, prefix: tokenPrefix(tokenizer, style: r.style, lyrics: r.lyrics, planning: r.planning, abcIDs: nil),
                                                            r.abcSampling, seed: r.seed, phase: .abc) { _ in
                        if isCancelled() { throw CancelledError() }
                        count += 1
                        if count % 16 == 0 {
                            let rate = Double(count) / Date().timeIntervalSince(start)
                            progress(Progress(stage: .planning, fraction: min(1, Double(count) / 2000), detail: String(format: "%d score tokens · %.0f tok/s", count, rate)))
                        }
                    }
                    abcText = tokenizer.decode(abcIDs)
                }
                if isInstrumental(r.style), let planned = abcText {
                    let (quiet, n) = silenceVocal(planned)
                    if n > 0 { abcText = quiet; abcIDs = tokenizer.encode(quiet); silenced = n }
                }
            }
            lap("plan")
            let prefix = tokenPrefix(tokenizer, style: r.style, lyrics: r.lyrics, planning: r.planning, abcIDs: abcIDs)

            // 2. semantic codec tokens
            let guidance = r.cfgScale ?? (r.planning == .off ? 1.01 : 1.0)
            let negative = guidance != 1 ? negativePrefix(tokenizer, planning: r.planning, abcIDs: abcIDs) : nil
            let start = Date(); var count = 0
            let maxTok = r.semanticSampling.max_tokens
            let (ids, semTrunc) = try generateTokens(model, prefix: prefix, r.semanticSampling, seed: r.seed, phase: .semantic,
                                                     negative: negative, cfgScale: guidance, legacyOff: r.planning == .off) { _ in
                if isCancelled() { throw CancelledError() }
                count += 1
                if count % 25 == 0 {
                    let rate = Double(count) / Date().timeIntervalSince(start)
                    progress(Progress(stage: .semantic, fraction: Double(count) / Double(maxTok),
                                      detail: String(format: "%@ of music · %.0f tok/s", fmtTime(Double(count) * secondsPerToken), rate)))
                }
            }
            lap("semantic")
            let codec = ids.map { $0 - CODEC_OFFSET }
            guard !codec.isEmpty else { throw PipelineError.message("The semantic stage produced no codec tokens") }

            // 3. acoustic latents
            let latents = try synthesize(model, prefix: prefix, codec: codec, seed: r.seed, steps: r.steps) { i, n in
                if isCancelled() { throw CancelledError() }
                progress(Progress(stage: .acoustic, fraction: Double(i) / Double(n), detail: "step \(i) of \(n)"))
            }
            eval(latents)
            lap("acoustic")
            return (latents, codec, abcText, abcTrunc, semTrunc)
        }()
        let codec = codecOut

        // 4. waveform
        if unloadModelForDecode { self.model = nil; Memory.clearCache() }
        else { Memory.clearCache() }
        progress(Progress(stage: .decoding, fraction: 0, detail: "Decoding audio"))
        let audio = vae.decodeTiled(latents, core: vaeCore) { f in
            progress(Progress(stage: .decoding, fraction: f, detail: "Decoding audio"))
        }
        let samples = audio.asType(.float32).asArray(Float.self)
        lap("decode")
        Memory.clearCache()
        return RenderResult(audio: samples, frames: codec.count, codec: codec, abc: abcText, abcTruncated: abcTrunc,
                            semanticTruncated: semTrunc, timings: timings,
                            peakMemory: Int((peaks.values.max() ?? 0) * 1e9), stagePeaksGB: peaks,
                            silencedGroups: silenced)
    }
}

public func fmtTime(_ s: Double) -> String { String(format: "%d:%02d", Int(s) / 60, Int(s) % 60) }

/// Debug: logits (float32) after the prefix and after each forced token, for parity checks against Python.
public func debugForcedLogits(_ model: Yue2Model, ids: [Int], prefix n: Int) -> [[Float]] {
    let caches = (0..<model.layerCount).map { _ in KVCache() }
    var logits = model.arStep(MLXArray(ids[..<n].map(Int32.init)).reshaped(1, -1), caches: caches)
    var out = [logits[0].asType(.float32).asArray(Float.self)]
    for t in ids[n...] {
        logits = model.arStep(MLXArray([Int32(t)]).reshaped(1, 1), caches: caches)
        out.append(logits[0].asType(.float32).asArray(Float.self))
    }
    return out
}

/// Debug: hidden state (last position, float32) after the embedding and after each AR layer for a prefill.
public func debugLayerStates(_ model: Yue2Model, ids: [Int]) -> [[Float]] {
    var x = model.model.embed(MLXArray(ids.map(Int32.init)).reshaped(1, -1))
    var out = [x[0, -1].asType(.float32).asArray(Float.self)]
    for layer in model.model.layers {
        x = layer.ar(x, cache: KVCache(), causal: true)
        out.append(x[0, -1].asType(.float32).asArray(Float.self))
    }
    return out
}
