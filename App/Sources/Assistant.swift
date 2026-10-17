import Foundation
import FoundationModels

/// Writing help from Apple's built-in on-device model (Foundation Models). The rules below were learned by
/// rendering and measuring YuE2 songs (README "What we learned"). Nothing leaves the device.
enum Assistant {
    static let system = "You help write songs for YuE2, a model that sings a style prompt plus lyrics. Follow the output "
        + "format exactly and output nothing else: no preamble, no markdown fences."

    static let lyricsRules = """
        Lyrics format YuE2 expects:
        - Sections on their own line in square brackets: [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge], [Outro].
        - [Intro] and [Outro] are usually left empty (instrumental). Blank line between sections.
        - For an instrumental, write only the section tags with nothing under them.
        - 4-8 short singable lines per section, consistent syllable counts within a section, natural rhymes.
        - Repeat the chorus text word for word each time it comes back.
        - Keep the whole song under about 4 minutes when sung (typically 2 verses, 3 choruses, 1 bridge).
        """

    static let styleRules = """
        Style prompt format YuE2 expects: ONE line of short comma-separated tags, like
        "English, indie folk-pop, warm male lead vocal, fingerpicked acoustic guitar, upright piano, soft brushed drums, 92 BPM, hopeful, intimate".
        Rules learned from testing:
        - Start with the language of the lyrics (English, Mandarin, ...).
        - ALWAYS include a literal vocal tag: "female vocal" or "male vocal". Words like "diva" alone flip the gender.
        - Exception: for an instrumental (no singing), use the tag "instrumental" instead of a vocal tag (not "no vocals", which made wordless vocals more likely), and do not start with a language.
        - Genre, 3-5 instruments, a BPM, 1-3 mood words. 10-16 tags in total.
        - Do not describe vocal technique (vibrato, belting, twang, breathy): the model ignores it.
        - Naming acoustic guitars or a gospel choir pulls songs toward country.
        """

    static var availability: String? {
        switch SystemLanguageModel.default.availability {
        case .available: nil
        case .unavailable(let reason): "Apple Intelligence model unavailable (\(reason))"
        }
    }

    static var variantName: String {
        SystemLanguageModel.default.variant == .coreAdvanced3 ? "AFM 3 Core Advanced" : "AFM 3 Core"
    }

    struct Reply { var text: String; var seconds: Double }

    static func run(_ prompt: String, instructions: String = system, temperature: Double = 0.8) async throws -> Reply {
        let start = Date()
        let session = LanguageModelSession(instructions: instructions)
        let r = try await session.respond(to: prompt, options: GenerationOptions(temperature: temperature))
        return Reply(text: unfence(r.content), seconds: Date().timeIntervalSince(start))
    }

    static func unfence(_ t: String) -> String {
        if let m = t.range(of: #"```[^\n]*\n([\s\S]*?)```"#, options: .regularExpression) {
            var s = String(t[m]); s = s.replacingOccurrences(of: #"^```[^\n]*\n"#, with: "", options: .regularExpression)
            return s.replacingOccurrences(of: "```", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func writeLyrics(brief: String, language: String, structure: String, style: String) async throws -> Reply {
        var p = "\(lyricsRules)\n\nWrite original song lyrics in \(language).\nBrief: \(brief)\n"
        if !structure.isEmpty { p += "Section order: \(structure)\n" }
        if !style.isEmpty { p += "Musical style (for feel only): \(style)\n" }
        p += "\nOutput only the lyrics with section tags."
        return try await run(p)
    }

    static func reviseLyrics(_ lyrics: String, instruction: String) async throws -> Reply {
        try await run("\(lyricsRules)\n\nRevise these lyrics. Instruction: \(instruction)\nKeep the section tags unless the "
                      + "instruction says otherwise.\n\nLYRICS:\n\(lyrics)\n\nOutput only the revised lyrics.")
    }

    static func writeStyle(brief: String, current: String, lyrics: String) async throws -> Reply {
        var p = "\(styleRules)\n\nWrite the style prompt for: \(brief.isEmpty ? "a song that suits these lyrics" : brief)\n"
        if !current.isEmpty { p += "Current style prompt (revise it): \(current)\n" }
        if !lyrics.isEmpty { p += "Lyrics:\n\(lyrics.prefix(2500))\n" }
        p += "\nOutput only the one line of tags."
        var r = try await run(p, temperature: 0.6)
        r.text = (r.text.split(separator: "\n").first.map(String.init) ?? r.text)
            .trimmingCharacters(in: CharacterSet(charactersIn: "\" ").union(.whitespaces))
        return r
    }

    static func ask(_ question: String, style: String, lyrics: String) async throws -> Reply {
        let ctx = [style.isEmpty ? "" : "STYLE: \(style)", lyrics.isEmpty ? "" : "LYRICS:\n\(lyrics.prefix(2500))"]
            .filter { !$0.isEmpty }.joined(separator: "\n\n")
        return try await run("\(styleRules)\n\n\(lyricsRules)\n\nThe user is making a song with YuE2.\n\(ctx)\n\n"
                             + "Question: \(question)\nAnswer concisely in plain text.",
                             instructions: "You are a helpful songwriting and music-production assistant.", temperature: 0.5)
    }
}
