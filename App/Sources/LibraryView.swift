import AVFoundation
import SwiftUI
import YuE2Kit

struct LibraryView: View {
    @Environment(Studio.self) private var studio
    @Binding var tab: Int

    var body: some View {
        NavigationStack {
            List {
                if studio.songs.isEmpty {
                    ContentUnavailableView("No songs yet", systemImage: "music.note",
                                           description: Text("Generate one on the Create tab."))
                }
                ForEach(studio.songs) { s in
                    HStack(spacing: 12) {
                        Button { studio.player.toggle(s.m4a) } label: {
                            Image(systemName: studio.player.isPlaying(s.m4a) ? "pause.circle.fill" : "play.circle.fill")
                                .font(.system(size: 34))
                        }.buttonStyle(.borderless)
                        NavigationLink(value: s) {
                            VStack(alignment: .leading) {
                                Text(s.name).font(.headline)
                                Text("\(fmtTime(s.seconds)) · \(s.date.formatted(date: .abbreviated, time: .shortened))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .onDelete { idx in idx.map { studio.songs[$0] }.forEach(studio.delete) }
            }
            .navigationTitle("Library")
            .navigationDestination(for: Song.self) { SongView(song: $0, tab: $tab) }
            .refreshable { studio.refreshSongs() }
            .onAppear { studio.refreshSongs() }
        }
    }
}

@MainActor @Observable
final class Player {
    var player: AVAudioPlayer?
    var current: URL?
    var playing = false
    var time = 0.0
    var error: String?
    private var timer: Timer?

    func load(_ url: URL) {
        guard current != url else { return }
        stop()
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        try? AVAudioSession.sharedInstance().setActive(true)
        do {
            player = try AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
            current = url; time = 0; error = nil
        } catch {
            self.error = error.localizedDescription; player = nil; current = nil
        }
    }

    /// Load (if needed) and play/pause this file.
    func toggle(_ url: URL) {
        load(url)
        toggle()
    }

    func isPlaying(_ url: URL) -> Bool { playing && current == url }

    func toggle() {
        guard let p = player else { return }
        if p.isPlaying { p.pause() } else { p.play() }
        playing = p.isPlaying
        timer?.invalidate()
        if playing {
            timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { _ in
                MainActor.assumeIsolated {
                    self.time = self.player?.currentTime ?? 0
                    if self.player?.isPlaying == false { self.playing = false }
                }
            }
        }
    }

    func seek(_ t: Double) { player?.currentTime = t; time = t }
    func stop() { player?.stop(); timer?.invalidate(); playing = false }
}

struct SongView: View {
    var song: Song
    @Binding var tab: Int
    @Environment(Studio.self) private var studio
    @State private var abcText: String?
    private var player: Player { studio.player }

    var body: some View {
        List {
            Section {
                HStack(spacing: 16) {
                    Button { player.toggle(song.m4a) } label: {
                        Image(systemName: player.isPlaying(song.m4a) ? "pause.circle.fill" : "play.circle.fill").font(.system(size: 52))
                    }.buttonStyle(.borderless)
                    VStack {
                        Slider(value: Binding(get: { player.time }, set: { player.seek($0) }),
                               in: 0...max(1, player.player?.duration ?? 1))
                        HStack {
                            Text(fmtTime(player.time)); Spacer(); Text(fmtTime(player.player?.duration ?? 0))
                        }.font(.caption).monospacedDigit().foregroundStyle(.secondary)
                    }
                }
            }
            if let e = player.error { Section { Text("Can't play: \(e)").foregroundStyle(.red) } }
            Section("Download") {
                ShareLink(item: song.m4a, preview: SharePreview("\(song.name).m4a")) { Label("Song (AAC, .m4a)", systemImage: "square.and.arrow.down") }
                ShareLink(item: song.wav, preview: SharePreview("\(song.name).wav")) { Label("Song (WAV, lossless)", systemImage: "waveform") }
                if FileManager.default.fileExists(atPath: song.abc.path) {
                    ShareLink(item: song.abc, preview: SharePreview("\(song.name).abc")) { Label("Score (.abc)", systemImage: "music.note.list") }
                }
            }
            Section {
                Text("iOS doesn't let apps add songs to the Music library directly. AirDrop the .m4a to your Mac and drop it into the Music app; with Sync Library on, it then appears in Music on this device too, with its title and cover art.")
                    .font(.footnote).foregroundStyle(.secondary)
            } header: { Text("Add to Apple Music") }
            Section("Reuse") {
                Button { studio.loadSettings(from: song); tab = 0 } label: { Label("Load these settings", systemImage: "slider.horizontal.3") }
                Button { studio.loadSettings(from: song); studio.seed = Int.random(in: 1...999_999); tab = 0 } label: {
                    Label("Same song, new seed", systemImage: "dice")
                }
                if FileManager.default.fileExists(atPath: song.abc.path) {
                    Button { studio.loadScore(from: song); tab = 0 } label: { Label("Use this score", systemImage: "music.note.list") }
                }
            }
            Section("Details") {
                ForEach(details, id: \.0) { k, v in LabeledContent(k, value: v) }
            }
            if let abcText {
                Section("Score") { Text(abcText).font(.system(.caption2, design: .monospaced)).textSelection(.enabled) }
            }
        }
        .navigationTitle(song.name)
        .onAppear {
            player.load(song.m4a)
            abcText = try? String(contentsOf: song.abc, encoding: .utf8)
        }
    }

    private var details: [(String, String)] {
        let i = song.info
        var out: [(String, String)] = []
        if let s = i["style"] as? String { out.append(("Style", s)) }
        if let v = i["seed"] as? Int { out.append(("Seed", "\(v)")) }
        if let v = i["planning"] as? String { out.append(("Planning", v)) }
        if let v = i["variant"] as? String { out.append(("Model", v)) }
        if let t = i["timings"] as? [String: Double] {
            let total = t.values.reduce(0, +)
            out.append(("Render time", fmtTime(total)))
            for k in ["load", "plan", "semantic", "acoustic", "decode"] {
                if let v = t[k], v > 0.05 { out.append(("  \(k)", String(format: "%.0f s", v))) }
            }
            if song.seconds > 0, total > 0 { out.append(("Speed", String(format: "%.2f× real time", song.seconds / total))) }
        }
        if let p = i["peak_gb"] as? Double { out.append(("Peak GPU memory", String(format: "%.2f GB", p))) }
        if let m = i["machine"] as? String { out.append(("Device", m)) }
        if let th = i["thermal_end"] as? String { out.append(("Thermal at end", th)) }
        if i["hit_max_tokens"] as? Bool == true { out.append(("Note", "Hit the length limit: the song may stop mid-line")) }
        return out
    }
}
