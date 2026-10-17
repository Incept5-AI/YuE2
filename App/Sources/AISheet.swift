import SwiftUI

/// The ✦ buttons: writing help from Apple's on-device model.
struct AISheet: View {
    enum Mode: String, Identifiable { case style, lyrics, revise, ask; var id: String { rawValue } }

    var mode: Mode
    @Environment(Studio.self) private var studio
    @Environment(\.dismiss) private var dismiss
    @State private var brief = ""
    @State private var language = "English"
    @State private var structure = "Intro, Verse, Chorus, Verse, Chorus, Bridge, Chorus, Outro"
    @State private var busy = false
    @State private var result: String?
    @State private var meta = ""
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                if let a = Assistant.availability {
                    Section { Label(a, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                }
                Section {
                    TextField(placeholder, text: $brief, axis: .vertical).lineLimit(2...6)
                    if mode == .lyrics {
                        Picker("Language", selection: $language) {
                            ForEach(["English", "French", "German", "Spanish", "Italian", "Mandarin"], id: \.self) { Text($0) }
                        }
                        TextField("Section order", text: $structure, axis: .vertical)
                    }
                    Button {
                        Task { await go() }
                    } label: {
                        HStack {
                            Label(busy ? "Thinking…" : "Ask the on-device model", systemImage: "sparkles")
                            if busy { Spacer(); ProgressView() }
                        }
                    }
                    .disabled(busy || (brief.isEmpty && mode != .style) || Assistant.availability != nil)
                } header: { Text(title) } footer: {
                    Text("Runs on \(Assistant.variantName), Apple's built-in model. Nothing leaves the device.")
                }
                if let error { Section { Text(error).foregroundStyle(.red) } }
                if let result {
                    Section {
                        Text(result).font(mode == .ask ? .body : .system(.callout, design: .monospaced)).textSelection(.enabled)
                    } header: { Text("Result") } footer: { Text(meta) }
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
                if let result, mode != .ask {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Use") {
                            if mode == .style { studio.style = result } else { studio.lyrics = result }
                            studio.editedSincePreset = true
                            dismiss()
                        }.bold()
                    }
                }
            }
        }
    }

    private var title: String {
        switch mode {
        case .style: "Write style"
        case .lyrics: "Write lyrics"
        case .revise: "Revise lyrics"
        case .ask: "Ask"
        }
    }

    private var placeholder: String {
        switch mode {
        case .style: "Describe the sound (optional), e.g. late-night jazz trio, male vocal"
        case .lyrics: "What is the song about?"
        case .revise: "e.g. make the chorus more hopeful; shorten verse 2"
        case .ask: "e.g. why does my song come out with the wrong voice?"
        }
    }

    private func go() async {
        busy = true; error = nil
        defer { busy = false }
        do {
            let r: Assistant.Reply
            switch mode {
            case .style: r = try await Assistant.writeStyle(brief: brief, current: studio.style, lyrics: studio.lyrics)
            case .lyrics: r = try await Assistant.writeLyrics(brief: brief, language: language, structure: structure, style: studio.style)
            case .revise: r = try await Assistant.reviseLyrics(studio.lyrics, instruction: brief)
            case .ask: r = try await Assistant.ask(brief, style: studio.style, lyrics: studio.lyrics)
            }
            result = r.text
            meta = String(format: "%.1f s on %@", r.seconds, Assistant.variantName)
        } catch {
            self.error = error.localizedDescription
        }
    }
}
