import Foundation

/// The frozen Qwen text/ABC byte-level BPE (tiktoken format), ported from generate.py's `Tokenizer`.
/// Only `encode_ordinary` is needed: prompts never contain special tokens.
public final class Tokenizer: @unchecked Sendable {
    private var ranks: [[UInt8]: Int] = [:]
    private var bytesFor: [[UInt8]] = []
    private var specials: [Int: String] = [:]
    private let regex: NSRegularExpression

    static let pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    public init(mergeFile: URL) throws {
        let text = try String(contentsOf: mergeFile, encoding: .utf8)
        var maxRank = 0
        var pairs: [([UInt8], Int)] = []
        pairs.reserveCapacity(152_000)
        for line in text.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: " ")
            guard parts.count == 2, let data = Data(base64Encoded: String(parts[0])), let r = Int(parts[1]) else { continue }
            pairs.append((Array(data), r))
            maxRank = max(maxRank, r)
        }
        bytesFor = Array(repeating: [], count: maxRank + 1)
        ranks.reserveCapacity(pairs.count)
        for (b, r) in pairs { ranks[b] = r; bytesFor[r] = b }
        var names = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<R>", "<S>", "<X>", "<mask>", "<sep>"]
        names += (0..<200).map { "<extra_\($0)>" }
        names[204] = "<abc>"; names[205] = "</abc>"
        for (i, n) in names.enumerated() { specials[i + pairs.count] = n }
        regex = try NSRegularExpression(pattern: Self.pattern)
    }

    public var vocabCount: Int { ranks.count + specials.count }

    public func encode(_ text: String) -> [Int] {
        let s = text.precomposedStringWithCanonicalMapping  // unicodedata.normalize("NFC", ...)
        let ns = s as NSString
        var out: [Int] = []
        for m in regex.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            let piece = Array(ns.substring(with: m.range).utf8)
            if let r = ranks[piece] { out.append(r) } else { out += bpe(piece) }
        }
        return out
    }

    /// tiktoken's byte_pair_merge: repeatedly merge the adjacent pair with the lowest rank.
    private func bpe(_ piece: [UInt8]) -> [Int] {
        var bounds = Array(0...piece.count)  // part i spans bounds[i]..<bounds[i+1]
        func rank(_ i: Int) -> Int {  // rank of merging parts i and i+1
            guard i + 2 < bounds.count else { return Int.max }
            return ranks[Array(piece[bounds[i]..<bounds[i + 2]])] ?? Int.max
        }
        var pr = (0..<(bounds.count - 1)).map(rank)
        while true {
            var best = Int.max, at = -1
            for (i, r) in pr.enumerated() where r < best { best = r; at = i }
            if at < 0 { break }
            bounds.remove(at: at + 1)
            pr.remove(at: at + 1)
            pr[at] = rank(at)
            if at > 0 { pr[at - 1] = rank(at - 1) }
        }
        return (0..<(bounds.count - 1)).map { ranks[Array(piece[bounds[$0]..<bounds[$0 + 1]])]! }
    }

    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            if id >= 0 && id < bytesFor.count && !bytesFor[id].isEmpty { bytes += bytesFor[id] }
            else if let s = specials[id] { bytes += Array(s.utf8) }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
