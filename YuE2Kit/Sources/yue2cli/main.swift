import Foundation
import MLX
import YuE2Kit

// Mac validation tool: renders with the Swift port and dumps token ids so they can be diffed against
// the Python runner (tools/reference.py) for the same seed.
//   yue2cli --model <dir> --style S --lyrics-file F [--cot full] [--seed N] [--abc-max N] [--sem-max N]
//           [--steps N] [--vae-core N] [--out song.wav] [--dump ids.json] [--tokenize-file F]

var args = [String: String]()
var it = CommandLine.arguments.dropFirst().makeIterator()
while let a = it.next() { if a.hasPrefix("--") { args[String(a.dropFirst(2))] = it.next() ?? "" } }

if let f = args["silence-abc"] {  // parity with tools/silence_vocal.py: one input file per line of a list
    for path in try String(contentsOfFile: f, encoding: .utf8).split(separator: "\n") {
        let (out, n) = silenceVocal(try String(contentsOfFile: String(path), encoding: .utf8))
        try out.write(toFile: String(path) + ".swift-silenced", atomically: true, encoding: .utf8)
        print(path, n)
    }
    exit(0)
}
if let f = args["tag-file"] {
    let art = args["artwork"].flatMap { try? Data(contentsOf: URL(fileURLWithPath: $0)) }
    try await tagM4A(URL(fileURLWithPath: f), title: args["title"] ?? "Test", genre: "Swing", comment: "test", artwork: art)
    print("tagged"); exit(0)
}
let modelDir = URL(fileURLWithPath: (args["model"]! as NSString).expandingTildeInPath)
if let f = args["tokenize-file"] {
    let tok = try Tokenizer(mergeFile: modelDir.appending(path: "qwen.tiktoken"))
    let text = try String(contentsOfFile: f, encoding: .utf8)
    let ids = tok.encode(text)
    print(String(data: try JSONSerialization.data(withJSONObject: ["ids": ids, "roundtrip": tok.decode(ids) == text.precomposedStringWithCanonicalMapping]), encoding: .utf8)!)
    exit(0)
}

if let f = args["logits-ids"] {
    // Teacher-forced check: prefill ids[..<n], then step through the rest; write float32 logits per step.
    let d = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: f))) as! [String: Any]
    let ids = d["ids"] as! [Int], n = d["prefix"] as! Int
    let model = try loadModel(directory: modelDir)
    var out = Data()
    let rows = args["layers"] != nil ? debugLayerStates(model, ids: Array(ids[..<n])) : debugForcedLogits(model, ids: ids, prefix: n)
    for l in rows { l.withUnsafeBytes { out.append(contentsOf: $0) } }
    try out.write(to: URL(fileURLWithPath: args["logits-out"]!))
    exit(0)
}

let t0 = Date()
let pipe = try Yue2Pipeline(directory: modelDir, lowMemory: args["low-memory"] != nil)
print(String(format: "loaded in %.1fs, active %.2f GB", Date().timeIntervalSince(t0), Double(Memory.activeMemory) / 1e9))
if let c = args["vae-core"] { pipe.vaeCore = Int(c)! }
var req = RenderRequest(style: args["style"] ?? "English, pop, female vocal, piano, 90 BPM",
                        lyrics: try String(contentsOfFile: args["lyrics-file"]!, encoding: .utf8))
req.planning = Planning(rawValue: args["cot"] ?? "full")!
req.seed = Int(args["seed"] ?? "1")!
req.steps = Int(args["steps"] ?? "32")!
if let m = args["abc-max"] { req.abcSampling.max_tokens = Int(m)! }
if let m = args["sem-max"] { req.semanticSampling.max_tokens = Int(m)! }
if let f = args["abc-file"] { req.abc = try String(contentsOfFile: f, encoding: .utf8) }

var lastStage = ""
let res = try pipe.render(req, progress: { p in
    if p.stage.rawValue != lastStage { print("[\(p.stage.rawValue)] \(p.detail)"); lastStage = p.stage.rawValue }
})
print("stage peaks GB", res.stagePeaksGB.mapValues { (($0 * 100).rounded() / 100) })
print("timings", res.timings.mapValues { (($0 * 10).rounded() / 10) })
print(String(format: "%d frames = %.1fs audio, peak %.2f GB", res.frames, Double(res.frames) * secondsPerToken, Double(res.peakMemory) / 1e9))
if let out = args["out"] { try writeWAV(res.audio, to: URL(fileURLWithPath: out)) }
if let dump = args["dump"] {
    let j: [String: Any] = ["abc": res.abc ?? "", "frames": res.frames, "codec": res.codec, "timings": res.timings]
    try JSONSerialization.data(withJSONObject: j, options: .prettyPrinted).write(to: URL(fileURLWithPath: dump))
}
