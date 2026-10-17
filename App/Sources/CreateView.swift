import SwiftUI
import YuE2Kit

struct CreateView: View {
    @Environment(Studio.self) private var studio
    @State private var aiSheet: AISheet.Mode?
    @State private var confirmPreset: Preset?
    @State private var showScore = false

    var body: some View {
        @Bindable var s = studio
        NavigationStack {
            Form {
                if studio.running != nil || studio.message != nil { JobCard() }

                Section {
                    Menu {
                        ForEach(groups, id: \.self) { g in
                            Section(g) {
                                ForEach(Preset.all.filter { $0.group == g }) { p in
                                    Button(p.label) { choose(p) }
                                }
                            }
                        }
                    } label: {
                        HStack {
                            Label(currentPreset?.label ?? "Choose a starting point", systemImage: currentPreset?.group.hasPrefix("Instrumental") == true ? "pianokeys" : "music.mic")
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down").foregroundStyle(.secondary)
                        }
                    }
                    if let p = currentPreset { Text(p.note).font(.footnote).foregroundStyle(.secondary) }
                } header: { Text("Start from a singer or instrumental") } footer: {
                    Text("Fills every setting below. Lyrics are originals written in each artist's style; instrumentals leave every section empty. Change anything afterwards.")
                }

                Section {
                    TextField("Comma-separated tags", text: $s.style, axis: .vertical)
                        .lineLimit(2...6)
                        .onChange(of: s.style) { s.editedSincePreset = true }
                    if !studio.hasVocalTag {
                        Label("Add a literal \"male vocal\" or \"female vocal\" tag", systemImage: "exclamationmark.triangle")
                            .font(.footnote).foregroundStyle(.orange)
                    }
                    if studio.isInstrumental && studio.planning != .full {
                        Label("Instrumentals need planning on Full, or the model sings", systemImage: "exclamationmark.triangle")
                            .font(.footnote).foregroundStyle(.orange)
                    }
                    Button { aiSheet = .style } label: { Label("Write style", systemImage: "sparkles") }
                } header: { Text("Style") }

                Section {
                    TextEditor(text: $s.lyrics)
                        .font(.system(.body, design: .monospaced))
                        .frame(minHeight: 240)
                        .onChange(of: s.lyrics) { s.editedSincePreset = true }
                    Text(lyricStats).font(.footnote).foregroundStyle(.secondary)
                    HStack {
                        Button { aiSheet = .lyrics } label: { Label("Write lyrics", systemImage: "sparkles") }
                        Spacer()
                        Button { aiSheet = .revise } label: { Label("Revise", systemImage: "wand.and.stars") }
                    }.buttonStyle(.borderless)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Outro]"], id: \.self) { tag in
                                Button(tag) { s.lyrics += (s.lyrics.hasSuffix("\n") || s.lyrics.isEmpty ? "" : "\n") + "\n\(tag)\n" }
                                    .buttonStyle(.bordered).font(.caption)
                            }
                        }
                    }
                } header: { Text("Lyrics") }

                Section {
                    Picker("Planning", selection: $s.planning) {
                        Text("Off").tag(Planning.off)
                        Text("Melody").tag(Planning.melody)
                        Text("Full").tag(Planning.full)
                    }.pickerStyle(.segmented)
                    Text(planningHelp).font(.footnote).foregroundStyle(.secondary)
                    if studio.planning != .off {
                        DisclosureGroup(isExpanded: $showScore) {
                            TextEditor(text: $s.abc)
                                .font(.system(.caption, design: .monospaced))
                                .frame(minHeight: 200)
                        } label: {
                            Text(studio.abc.isEmpty ? "No score: the model will write one" : "Score: \(studio.abc.count) characters")
                        }
                        HStack {
                            Button { studio.plan() } label: { Label("Plan score", systemImage: "music.note.list") }
                                .disabled(studio.running != nil)
                            Spacer()
                            Button("Clear", role: .destructive) { s.abc = "" }.disabled(studio.abc.isEmpty)
                        }.buttonStyle(.borderless)
                    }
                } header: { Text("Score (ABC)") }

                Section {
                    TextField("Song name", text: $s.name).textInputAutocapitalization(.never).autocorrectionDisabled()
                    HStack {
                        Text("Seed")
                        TextField("Seed", value: $s.seed, format: .number.grouping(.never))
                            .keyboardType(.numberPad).multilineTextAlignment(.trailing)
                            .disabled(studio.randomSeed)
                        Button { s.seed = Int.random(in: 1...999_999) } label: { Image(systemName: "dice") }
                            .buttonStyle(.borderless)
                    }
                    Toggle("New random seed each time", isOn: $s.randomSeed)
                    LabeledSlider(title: "Max length", value: $s.lengthSeconds, range: 20...360, step: 10,
                                  text: fmtTime(studio.lengthSeconds))
                    Toggle("Automatic guidance (CFG)", isOn: $s.cfgAuto)
                    if !studio.cfgAuto {
                        LabeledSlider(title: "CFG", value: $s.cfg, range: 1.0...3.0, step: 0.05, text: String(format: "%.2f", studio.cfg))
                    }
                    LabeledSlider(title: "Flow steps", value: $s.steps, range: 8...64, step: 4, text: "\(Int(studio.steps))")
                    DisclosureGroup("Sampling") {
                        SamplingEditor(title: "Music tokens", sampling: $s.semanticSampling)
                        SamplingEditor(title: "Score planning", sampling: $s.abcSampling)
                        Button("Reset to model defaults") {
                            s.semanticSampling = .semanticDefault; s.abcSampling = .abcDefault
                        }
                    }
                } header: { Text("Generation") } footer: {
                    Text("Songs usually run about 25% longer than the score. Fewer flow steps is faster with some loss of detail.")
                }

                Section {
                    Button(action: studio.generate) {
                        Label(studio.running == "render" ? "Generating…" : "Generate song", systemImage: "waveform")
                            .frame(maxWidth: .infinity).font(.headline)
                    }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(studio.running != nil || studio.lyrics.isEmpty || studio.installedVariant == nil)
                    if studio.installedVariant == nil {
                        Text("Download the model in the Model tab first.").font(.footnote).foregroundStyle(.orange)
                    }
                }
            }
            .navigationTitle("YuE2 Studio")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { aiSheet = .ask } label: { Label("Ask", systemImage: "questionmark.bubble") }
                }
            }
            .sheet(item: $aiSheet) { mode in AISheet(mode: mode) }
            .confirmationDialog("Replace your edits with this preset?", isPresented: .constant(confirmPreset != nil),
                                titleVisibility: .visible, presenting: confirmPreset) { p in
                Button("Use \(p.label)", role: .destructive) { studio.apply(p); confirmPreset = nil }
                Button("Keep my edits", role: .cancel) { confirmPreset = nil }
            }
        }
    }

    private var groups: [String] {
        var seen: [String] = []
        for p in Preset.all where !seen.contains(p.group) { seen.append(p.group) }
        return seen
    }

    private var currentPreset: Preset? { Preset.all.first { $0.id == studio.presetID } }

    private func choose(_ p: Preset) {
        if studio.editedSincePreset { confirmPreset = p } else { studio.apply(p) }
    }

    private var lyricStats: String {
        let lines = studio.lyrics.split(separator: "\n")
        let sung = lines.filter { !$0.hasPrefix("[") && !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let sections = lines.filter { $0.hasPrefix("[") }.count
        let words = sung.reduce(0) { $0 + $1.split(separator: " ").count }
        if sung.isEmpty && sections > 0 { return "\(sections) empty sections: instrumental" }
        return "\(sections) sections · \(sung.count) lines · \(words) words"
    }

    private var planningHelp: String {
        switch studio.planning {
        case .off: "No score: fastest, least structured."
        case .melody: "Melody-only score, no chords (the model card's recipe for covers)."
        case .full: "Chord-annotated score first, then the music follows it (best adherence)."
        }
    }
}

struct LabeledSlider: View {
    var title: String
    @Binding var value: Double
    var range: ClosedRange<Double>
    var step: Double
    var text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack { Text(title); Spacer(); Text(text).monospacedDigit().foregroundStyle(.secondary) }
            Slider(value: $value, in: range, step: step)
        }
    }
}

struct SamplingEditor: View {
    var title: String
    @Binding var sampling: Sampling

    var body: some View {
        VStack(alignment: .leading) {
            Text(title).font(.subheadline.bold()).padding(.top, 4)
            LabeledSlider(title: "Temperature", value: floatBinding(\.temperature), range: 0.1...1.5, step: 0.05,
                          text: String(format: "%.2f", sampling.temperature))
            LabeledSlider(title: "Top-p", value: floatBinding(\.top_p), range: 0.5...1.0, step: 0.01,
                          text: String(format: "%.2f", sampling.top_p))
            LabeledSlider(title: "Top-k", value: intBinding(\.top_k), range: 5...300, step: 5, text: "\(sampling.top_k)")
            LabeledSlider(title: "Repetition penalty", value: floatBinding(\.repetition_penalty), range: 1.0...1.5, step: 0.005,
                          text: String(format: "%.3f", sampling.repetition_penalty))
        }
    }

    private func floatBinding(_ k: WritableKeyPath<Sampling, Float>) -> Binding<Double> {
        Binding(get: { Double(sampling[keyPath: k]) }, set: { sampling[keyPath: k] = Float($0) })
    }

    private func intBinding(_ k: WritableKeyPath<Sampling, Int>) -> Binding<Double> {
        Binding(get: { Double(sampling[keyPath: k]) }, set: { sampling[keyPath: k] = Int($0) })
    }
}

struct JobCard: View {
    @Environment(Studio.self) private var studio

    var body: some View {
        Section {
            if let p = studio.progress {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(stageName(p.stage)).font(.headline)
                        Spacer()
                        if let t = studio.startedAt {
                            Text(t, style: .timer).monospacedDigit().foregroundStyle(.secondary)
                        }
                    }
                    ProgressView(value: studio.overallFraction)
                    Text(p.detail).font(.footnote).foregroundStyle(.secondary)
                    if studio.pausedInBackground {
                        Text("Paused while the app was in the background").font(.footnote).foregroundStyle(.orange)
                    }
                    Text("Keep YuE2 Studio open: iOS stops GPU work in the background.").font(.caption2).foregroundStyle(.secondary)
                    Button("Cancel", role: .destructive) { studio.cancel() }.buttonStyle(.borderless)
                }
            } else if let m = studio.message {
                HStack {
                    Text(m).font(.callout)
                    Spacer()
                    Button { studio.message = nil } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                }
                if let name = studio.lastSong, let song = studio.songs.first(where: { $0.name == name }) {
                    HStack(spacing: 14) {
                        Button { studio.player.toggle(song.m4a) } label: {
                            Image(systemName: studio.player.isPlaying(song.m4a) ? "pause.circle.fill" : "play.circle.fill")
                                .font(.system(size: 44))
                        }.buttonStyle(.borderless)
                        VStack(alignment: .leading) {
                            Text(song.name).font(.headline)
                            Text("\(fmtTime(song.seconds)) · also in the Library tab").font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        ShareLink(item: song.m4a) { Image(systemName: "square.and.arrow.up") }.buttonStyle(.borderless)
                    }
                }
            }
        }
    }

    private func stageName(_ s: Stage) -> String {
        switch s {
        case .loading: "Loading model"
        case .planning: "Writing the score"
        case .semantic: "Composing (music tokens)"
        case .acoustic: "Rendering sound (flow matching)"
        case .decoding: "Decoding audio"
        case .done: "Done"
        }
    }
}
