import Foundation
import Observation
import os
import UIKit
import YuE2Kit

struct Preset: Codable, Identifiable, Hashable {
    var id: String
    var label: String
    var group: String
    var seed: Int
    var style: String
    var lyrics: String
    var note: String
    var cot: String
    var cfg_scale: Double?
    var steps: Int
    var length_s: Int
    var abc: String

    static let all: [Preset] = {
        guard let url = Bundle.main.url(forResource: "presets", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([Preset].self, from: data)) ?? []
    }()
    static let defaultID = "buble"
}

struct Song: Identifiable, Hashable {
    var name: String
    var folder: URL
    var date: Date
    var info: [String: Any]
    var id: String { name }
    static func == (a: Song, b: Song) -> Bool { a.name == b.name && a.date == b.date }
    func hash(into h: inout Hasher) { h.combine(name) }

    var m4a: URL { folder.appending(path: "song.m4a") }
    var wav: URL { folder.appending(path: "song.wav") }
    var abc: URL { folder.appending(path: "score.abc") }
    var request: URL { folder.appending(path: "request.json") }
    var seconds: Double { info["seconds"] as? Double ?? 0 }
}

/// Blocks the worker between GPU evaluations while the app is in the background (iOS forbids GPU work
/// there), and carries the cancel flag. Checked once per token / flow step / decode tile.
final class Gate: @unchecked Sendable {
    private let cond = NSCondition()
    private var _cancelled = false, _paused = false
    var cancelled: Bool { cond.lock(); defer { cond.unlock() }; return _cancelled }
    func cancel() { cond.lock(); _cancelled = true; cond.broadcast(); cond.unlock() }
    func setPaused(_ p: Bool) { cond.lock(); _paused = p; cond.broadcast(); cond.unlock() }
    /// Returns true if the job should stop.
    func checkpoint() -> Bool {
        cond.lock(); defer { cond.unlock() }
        while _paused && !_cancelled { cond.wait() }
        return _cancelled
    }
}

/// Owns the MLX pipeline. Used from one detached task at a time (the UI allows a single job).
final class Worker: @unchecked Sendable {
    private(set) var pipeline: Yue2Pipeline?
    private(set) var variant: String?

    func pipeline(for v: String) throws -> Yue2Pipeline {
        if let pipeline, variant == v { return pipeline }
        pipeline = nil; variant = nil
        Memory.clearCache()
        // iOS kills (SIGKILL, no error) an app that exceeds its memory budget, so refuse up front instead.
        let needGB = Double(ModelStore.sizeOnDisk(v)) / 1e9 + 1.0
        let haveGB = Double(os_proc_available_memory()) / 1e9
        if haveGB < needGB {
            throw PipelineErrorUI(String(format: "Not enough memory for the %@ model: iOS allows this app %.1f GB and it needs about %.1f GB. "
                + "The build needs the increased-memory-limit entitlement; on 12 GB devices try 4-bit.", v, haveGB, needGB))
        }
        // Phones get ~6.4 GB with the increased-memory entitlement (iPhone 18 Pro), the M5 iPad ~17 GB.
        let p = try Yue2Pipeline(directory: ModelStore.directory(v), lowMemory: haveGB < 9)
        pipeline = p; variant = v
        return p
    }

    func unload() { pipeline = nil; variant = nil; Memory.clearCache() }
}

@MainActor @Observable
final class Studio {
    // Song settings (all editable after a preset fills them)
    var presetID = Preset.defaultID
    var name = "buble"
    var style = ""
    var lyrics = ""
    var abc = ""
    var planning: Planning = .full
    var seed = 19590
    var randomSeed = false
    var lengthSeconds = 300.0
    var cfgAuto = true
    var cfg = 1.5
    var steps = 32.0
    var abcSampling = Sampling.abcDefault
    var semanticSampling = Sampling.semanticDefault
    var variant = "4bit"
    var editedSincePreset = false

    // Job state
    var running: String?          // "render" | "plan"
    var progress: YuE2Kit.Progress?
    var startedAt: Date?
    var stageTimes: [String: Double] = [:]
    var message: String?
    var lastSong: String?
    var songs: [Song] = []
    var pausedInBackground = false

    let store = ModelStore()
    let player = Player()
    private let worker = Worker()
    private var gate = Gate()

    init() {
        if let p = Preset.all.first(where: { $0.id == Preset.defaultID }) ?? Preset.all.first { apply(p) }
        refreshSongs()
    }

    /// Test hook for runs driven from the Mac (`devicectl device process launch --environment-variables`):
    /// YUE2_AUTORUN=<preset id> renders that preset on launch; YUE2_SECONDS caps its length. Logs to stdout.
    func autorunIfRequested() {
        let env = ProcessInfo.processInfo.environment
        guard let id = env["YUE2_AUTORUN"], let p = Preset.all.first(where: { $0.id == id }) else { return }
        apply(p)
        if let sec = env["YUE2_SECONDS"].flatMap(Double.init) { lengthSeconds = sec }
        if let st = env["YUE2_STEPS"].flatMap(Double.init) { steps = st }
        if let v = env["YUE2_VARIANT"], ModelStore.isInstalled(v) { variant = v }
        name = env["YUE2_NAME"] ?? "\(p.id)-\(Self.machine.lowercased().replacingOccurrences(of: ",", with: "-"))"
        print(String(format: "YUE2 autorun %@ on %@: RAM %.1f GB, budget %.2f GB, thermal %@", id, Self.machine,
                     DeviceInfo.physicalGB, DeviceInfo.availableGB, DeviceInfo.thermal))
        logProgress = true
        generate()
    }
    private var logProgress = false
    private var lastLogged = ""

    private func log(_ p: YuE2Kit.Progress) {
        guard logProgress else { return }
        let key = "\(p.stage.rawValue)-\(Int(p.fraction * 10))"
        guard key != lastLogged else { return }
        lastLogged = key
        print(String(format: "YUE2 %6.1fs %@ %3.0f%% %@ | budget %.2f GB, MLX active %.2f GB, thermal %@",
                     Date().timeIntervalSince(startedAt ?? Date()), p.stage.rawValue, p.fraction * 100, p.detail,
                     DeviceInfo.availableGB, Double(Memory.activeMemory) / 1e9, DeviceInfo.thermal))
    }

    nonisolated static var songsRoot: URL { URL.documentsDirectory.appending(path: "Songs", directoryHint: .isDirectory) }

    func apply(_ p: Preset) {
        presetID = p.id
        style = p.style; lyrics = p.lyrics; abc = p.abc
        planning = Planning(rawValue: p.cot) ?? .full
        seed = p.seed; randomSeed = false
        cfgAuto = p.cfg_scale == nil; cfg = p.cfg_scale ?? 1.5
        steps = Double(p.steps)
        lengthSeconds = Double(p.length_s)
        abcSampling = .abcDefault; semanticSampling = .semanticDefault
        name = p.id
        editedSincePreset = false
    }

    var installedVariant: String? {
        ModelStore.isInstalled(variant) ? variant : ModelStore.installed.first
    }

    var hasVocalTag: Bool {
        isInstrumental || style.range(of: #"\b(fe)?male\b.*\bvocals?\b"#, options: [.regularExpression, .caseInsensitive]) != nil
    }

    /// An "instrumental" tag. Needs empty lyric sections and Full planning (see README "Instrumentals").
    var isInstrumental: Bool { YuE2Kit.isInstrumental(style) }

    /// Rough fraction of the whole job, weighting stages by their measured share of time.
    var overallFraction: Double {
        guard let p = progress else { return 0 }
        let order: [(Stage, Double)] = [(.loading, 0.03), (.planning, 0.12), (.semantic, 0.35), (.acoustic, 0.45), (.decoding, 0.05)]
        var acc = 0.0
        for (s, w) in order {
            if s == p.stage { return min(1, acc + w * p.fraction) }
            acc += w
        }
        return 1
    }

    // MARK: Jobs

    func setBackground(_ bg: Bool) {
        pausedInBackground = bg && running != nil
        gate.setPaused(bg)
    }

    func cancel() { gate.cancel() }

    private func uniqueName(_ base: String) -> String {
        let clean = base.lowercased().replacingOccurrences(of: #"[^a-z0-9\-]+"#, with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let stem = clean.isEmpty ? "song" : clean
        var n = stem, i = 2
        while FileManager.default.fileExists(atPath: Self.songsRoot.appending(path: n).path) { n = "\(stem)-\(i)"; i += 1 }
        return n
    }

    func makeRequest() -> RenderRequest {
        var r = RenderRequest(style: style, lyrics: lyrics)
        r.planning = planning
        if randomSeed { seed = Int.random(in: 1...999_999) }
        r.seed = seed
        let a = abc.trimmingCharacters(in: .whitespacesAndNewlines)
        r.abc = (planning != .off && !a.isEmpty) ? abc : nil
        r.cfgScale = cfgAuto ? nil : Float(cfg)
        r.steps = Int(steps)
        r.abcSampling = abcSampling
        var s = semanticSampling
        s.max_tokens = min(9000, Int(lengthSeconds / secondsPerToken))
        s.min_tokens = min(s.min_tokens, s.max_tokens)
        r.semanticSampling = s
        return r
    }

    func generate() {
        guard running == nil else { return }
        guard let v = installedVariant else { message = "Download the model first (Model tab)."; return }
        let req = makeRequest()
        let songName = uniqueName(name)
        start("render")
        let gate = self.gate, worker = self.worker
        let availStart = DeviceInfo.availableGB
        Task.detached(priority: .userInitiated) {
            do {
                let pipe = try worker.pipeline(for: v)
                let res = try pipe.render(req, isCancelled: { gate.checkpoint() }, progress: { p in
                    _ = gate.checkpoint()
                    Task { @MainActor in self.progress = p; self.log(p) }
                })
                await MainActor.run { self.progress = YuE2Kit.Progress(stage: .decoding, fraction: 1, detail: "Saving") }
                let folder = Self.songsRoot.appending(path: songName, directoryHint: .isDirectory)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try writeM4A(res.audio, to: folder.appending(path: "song.m4a"))
                // Title, artist, album, genre and cover art, so the file looks right once imported into Music.
                // The first tag is the language for songs, the genre for instrumentals.
                let genre = req.style.split(separator: ",").dropFirst(YuE2Kit.isInstrumental(req.style) ? 0 : 1).first.map { $0.trimmingCharacters(in: .whitespaces).capitalized }
                let art = Bundle.main.url(forResource: "artwork", withExtension: "png").flatMap { try? Data(contentsOf: $0) }
                try? await tagM4A(folder.appending(path: "song.m4a"), title: Self.prettyTitle(songName), genre: genre,
                                  comment: "Seed \(req.seed) · \(req.style)", artwork: art)
                try writeWAV(res.audio, to: folder.appending(path: "song.wav"))
                if let abc = res.abc { try abc.write(to: folder.appending(path: "score.abc"), atomically: true, encoding: .utf8) }
                let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                try enc.encode(req).write(to: folder.appending(path: "request.json"))
                let info: [String: Any] = [
                    "name": songName, "variant": v, "seconds": Double(res.frames) * secondsPerToken,
                    "frames": res.frames, "timings": res.timings, "stage_peaks_gb": res.stagePeaksGB, "low_memory": pipe.unloadModelForDecode, "instrumental_groups_silenced": res.silencedGroups, "peak_gb": Double(res.peakMemory) / 1e9,
                    "available_gb_at_start": availStart, "device": UIDevice.current.model,
                    "machine": Self.machine, "thermal_end": DeviceInfo.thermal,
                    "abc_truncated": res.abcTruncated, "hit_max_tokens": res.semanticTruncated,
                    "style": req.style, "seed": req.seed, "planning": req.planning.rawValue,
                    "created": ISO8601DateFormatter().string(from: Date()),
                ]
                try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
                    .write(to: folder.appending(path: "result.json"))
                await MainActor.run {
                    let total = res.timings.values.reduce(0, +)
                    self.message = String(format: "Saved %@: %@ of audio in %@%@", songName,
                                          fmtTime(Double(res.frames) * secondsPerToken), fmtTime(total),
                                          res.semanticTruncated ? " (hit the length limit)" : "")
                    self.lastSong = songName
                    if self.logProgress { print("YUE2 DONE \(self.message ?? "") | " + (String(data: (try? JSONSerialization.data(withJSONObject: info, options: [.sortedKeys])) ?? Data(), encoding: .utf8) ?? "")) }
                    self.finish()
                }
            } catch {
                await MainActor.run {
                    self.message = error is CancelledError ? "Cancelled" : "Failed: \(error.localizedDescription)"
                    if self.logProgress { print("YUE2 FAILED \(self.message ?? "")") }
                    self.finish()
                }
            }
        }
    }

    func plan() {
        guard running == nil else { return }
        guard planning != .off else { message = "Set planning to melody or full to write a score."; return }
        guard let v = installedVariant else { message = "Download the model first (Model tab)."; return }
        if randomSeed { seed = Int.random(in: 1...999_999) }
        let (st, ly, pl, sd, smp) = (style, lyrics, planning, seed, abcSampling)
        start("plan")
        let gate = self.gate, worker = self.worker
        Task.detached(priority: .userInitiated) {
            do {
                let pipe = try worker.pipeline(for: v)
                let (text, trunc) = try pipe.plan(style: st, lyrics: ly, planning: pl, seed: sd, sampling: smp,
                                                  isCancelled: { gate.checkpoint() }) { p in Task { @MainActor in self.progress = p } }
                await MainActor.run {
                    self.abc = text
                    self.message = trunc ? "Score written but it hit the token limit" : "Score written: it will be used for the next render"
                    self.finish()
                }
            } catch {
                await MainActor.run {
                    self.message = error is CancelledError ? "Cancelled" : "Failed: \(error.localizedDescription)"
                    self.finish()
                }
            }
        }
    }

    func unloadModel() {
        guard running == nil else { return }
        worker.unload()
        message = "Model unloaded from memory"
    }

    private func start(_ kind: String) {
        gate = Gate()
        running = kind; startedAt = Date(); message = nil
        progress = YuE2Kit.Progress(stage: .loading, fraction: 0, detail: worker.pipeline == nil ? "Loading model" : "Starting")
        UIApplication.shared.isIdleTimerDisabled = true
    }

    private func finish() {
        running = nil; progress = nil; startedAt = nil; pausedInBackground = false
        UIApplication.shared.isIdleTimerDisabled = false
        refreshSongs()
    }

    // MARK: Library

    func refreshSongs() {
        let fm = FileManager.default
        let dirs = (try? fm.contentsOfDirectory(at: Self.songsRoot, includingPropertiesForKeys: [.creationDateKey])) ?? []
        songs = dirs.compactMap { d in
            guard fm.fileExists(atPath: d.appending(path: "song.m4a").path) else { return nil }
            let info = (try? JSONSerialization.jsonObject(with: Data(contentsOf: d.appending(path: "result.json")))) as? [String: Any] ?? [:]
            let date = (try? d.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return Song(name: d.lastPathComponent, folder: d, date: date, info: info)
        }.sorted { $0.date > $1.date }
        print("YUE2 library: \(songs.count) songs of \(dirs.count) folders in \(Self.songsRoot.path): \(dirs.map(\.lastPathComponent))")
    }

    func delete(_ s: Song) {
        try? FileManager.default.removeItem(at: s.folder)
        refreshSongs()
    }

    func loadSettings(from s: Song) {
        guard let data = try? Data(contentsOf: s.request), let r = try? JSONDecoder().decode(RenderRequest.self, from: data) else { return }
        style = r.style; lyrics = r.lyrics; planning = r.planning; seed = r.seed; randomSeed = false
        abc = r.abc ?? ""
        cfgAuto = r.cfgScale == nil; cfg = Double(r.cfgScale ?? 1.5); steps = Double(r.steps)
        abcSampling = r.abcSampling; semanticSampling = r.semanticSampling
        lengthSeconds = Double(r.semanticSampling.max_tokens) * secondsPerToken
        semanticSampling.max_tokens = Sampling.semanticDefault.max_tokens
        name = s.name
        editedSincePreset = true
    }

    func loadScore(from s: Song) {
        if let t = try? String(contentsOf: s.abc, encoding: .utf8) {
            abc = t
            if planning == .off { planning = .full }
        }
    }

    nonisolated static func prettyTitle(_ name: String) -> String {
        name.replacingOccurrences(of: "-", with: " ").capitalized
    }

    nonisolated static let machine: String = {
        var u = utsname(); uname(&u)
        return withUnsafeBytes(of: &u.machine) { String(decoding: $0.prefix(while: { $0 != 0 }), as: UTF8.self) }
    }()
}
