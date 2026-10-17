import Foundation
import MLX
import MLXNN

// Port of yue2_vae.py: the Oobleck VAE decoder (channels-last; weight norm folded by convert.py).

public let frameSamples = 1920  // samples per latent frame at 48 kHz

final class Snake: Module, UnaryLayer {
    var alpha: MLXArray
    var beta: MLXArray
    init(_ ch: Int) { alpha = MLXArray.zeros([ch]); beta = MLXArray.zeros([ch]) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha), b = exp(beta)
        return x + (1.0 / (b + 1e-9)) * pow(sin(x * a), 2)
    }
}

final class ResidualUnit: Module, UnaryLayer {
    @ModuleInfo var layers: [UnaryLayer]
    init(_ ch: Int, dilation: Int) {
        _layers.wrappedValue = [
            Snake(ch), Conv1d(inputChannels: ch, outputChannels: ch, kernelSize: 7, padding: 3 * dilation, dilation: dilation),
            Snake(ch), Conv1d(inputChannels: ch, outputChannels: ch, kernelSize: 1),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for l in layers { y = l(y) }
        return x + y
    }
}

final class DecoderBlock: Module, UnaryLayer {
    @ModuleInfo var layers: [UnaryLayer]
    init(_ cin: Int, _ cout: Int, stride: Int) {
        _layers.wrappedValue = [
            Snake(cin),
            ConvTransposed1d(inputChannels: cin, outputChannels: cout, kernelSize: 2 * stride, stride: stride,
                             padding: Int((Double(stride) / 2).rounded(.up))),
            ResidualUnit(cout, dilation: 1), ResidualUnit(cout, dilation: 3), ResidualUnit(cout, dilation: 9),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for l in layers { y = l(y) }
        return y
    }
}

struct VAEConfig: Codable {
    struct Decoder: Codable {
        var channels: Int
        var c_mults: [Int]
        var strides: [Int]
        var latent_dim: Int
        var out_channels: Int
    }
    var decoder_config: Decoder
}

public final class OobleckDecoder: Module {
    @ModuleInfo var layers: [UnaryLayer]

    init(_ cfg: VAEConfig.Decoder) {
        let c = cfg.channels, mults = [1] + cfg.c_mults, strides = cfg.strides
        var ls: [UnaryLayer] = [Conv1d(inputChannels: cfg.latent_dim, outputChannels: mults.last! * c, kernelSize: 7, padding: 3)]
        for i in stride(from: mults.count - 1, to: 0, by: -1) {
            ls.append(DecoderBlock(mults[i] * c, mults[i - 1] * c, stride: strides[i - 1]))
        }
        ls += [Snake(mults[0] * c), Conv1d(inputChannels: mults[0] * c, outputChannels: cfg.out_channels, kernelSize: 7, padding: 3, bias: false)]
        _layers.wrappedValue = ls
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for l in layers { y = l(y) }
        return y
    }

    /// z [T,64] → audio [1920*T-64, 2] float32, exact tile cores, no crossfade.
    /// `core` 128 is bit-identical to the original 1024 and peaks at ~3.4 GB instead of ~5.9 GB (measured on the Mac).
    public func decodeTiled(_ z: MLXArray, core: Int = 128, halo: Int = 16, progress: ((Double) -> Void)? = nil) -> MLXArray {
        let frames = z.dim(0), total = frameSamples * frames - 64
        var out: [MLXArray] = []
        for start in stride(from: 0, to: frames, by: core) {
            let end = min(frames, start + core)
            let left = max(0, start - halo), right = min(frames, end + halo)
            let tile = self(z[left..<right][.newAxis])[0]
            let crop = (start - left) * frameSamples
            let len = min(end * frameSamples, total) - start * frameSamples
            let piece = clip(tile[crop..<(crop + len)], min: -1, max: 1)
            eval(piece)
            out.append(piece)
            Memory.clearCache()
            progress?(Double(end) / Double(frames))
        }
        return concatenated(out)
    }
}

public func loadVAE(directory: URL) throws -> OobleckDecoder {
    let cfg = try JSONDecoder().decode(VAEConfig.self, from: Data(contentsOf: directory.appending(path: "vae_config.json")))
    let vae = OobleckDecoder(cfg.decoder_config)
    let w = try loadArrays(url: directory.appending(path: "vae.safetensors"))
    try vae.update(parameters: ModuleParameters.unflattened(w), verify: [.all])
    eval(vae.parameters())
    return vae
}
