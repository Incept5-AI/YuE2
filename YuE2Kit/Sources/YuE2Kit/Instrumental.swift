import Foundation

/// A literal "instrumental" tag in the style prompt.
public func isInstrumental(_ style: String) -> Bool {
    style.split(separator: ",").contains { $0.trimmingCharacters(in: .whitespaces).lowercased() == "instrumental" }
}

/// Make a planned score instrumental: every note in the Vocal line becomes a rest of the same length (chord
/// symbols stay), and where the Ins line of that group is silent the vocal tune moves there instead.
///
/// With empty lyrics and an "instrumental" tag the planner usually writes an all-rest Vocal line. On some seeds
/// it writes a vocal melody anyway, and then the model sings it (3 of 7 presets, 2026-10-17). Port of
/// `silence_vocal` in `tools/silence_vocal.py`; `yue2cli --silence-abc` checks the two give identical output.
/// Returns the score and the number of groups changed.
public func silenceVocal(_ abc: String) -> (String, Int) {
    var lines = abc.components(separatedBy: "\n")
    if lines.last == "" { lines.removeLast() }
    var changed = 0, i = 0
    var prevIns: Int? = nil
    func skipFields(_ k: Int) -> Int {
        var k = k
        while k < lines.count && (lines[k].hasPrefix("M:") || lines[k].hasPrefix("K:")) { k += 1 }
        return k
    }
    while i < lines.count {
        guard lines[i] == "V: Vocal" else { i += 1; continue }
        let v = skipFields(i + 1)
        var n = v + 1
        while n < lines.count && lines[n] != "V: Ins" { n += 1 }
        let ins = skipFields(n + 1)
        if v >= lines.count || ins >= lines.count { break }
        let vocal = lines[v]
        if hasNotes(vocal) {
            let tune = vocal.replacingOccurrences(of: #""[^"\n]*""#, with: "", options: .regularExpression)
            lines[v] = mergeRests(restify(vocal))
            if !hasNotes(lines[ins]) {
                // A tie could run into a silent Ins group, or out of the previous Ins line into this one.
                lines[ins] = tune.replacingOccurrences(of: #"(?<=[A-Ga-g,'0-9])-"#, with: "", options: .regularExpression)
                if let p = prevIns {
                    lines[p] = lines[p].replacingOccurrences(of: #"-(\|)$"#, with: "$1", options: .regularExpression)
                }
            }
            changed += 1
        }
        prevIns = ins
        i = ins + 1
    }
    let out = lines.joined(separator: "\n") + (abc.hasSuffix("\n") ? "\n" : "")
    return (out, changed)
}

// One ABC token: chord symbol, inline key, or note/rest with accidental, octave marks, length and tie.
private let token = try! NSRegularExpression(pattern: #""[^"\n]*"|\[K:[^\]\n]+\]|(?:\^\^|__|\^|_|=)?([A-Ga-gz])[,']*([0-9]*)-?"#)

private func matches(_ s: String) -> [NSTextCheckingResult] {
    token.matches(in: s, range: NSRange(s.startIndex..., in: s))
}

private func group(_ m: NSTextCheckingResult, _ k: Int, in s: String) -> String? {
    Range(m.range(at: k), in: s).map { String(s[$0]) }
}

private func hasNotes(_ s: String) -> Bool {
    matches(s).contains { m in group(m, 1, in: s).map { $0 != "z" } ?? false }
}

private func restify(_ s: String) -> String {
    var out = "", last = s.startIndex
    for m in matches(s) {
        let r = Range(m.range, in: s)!
        out += s[last..<r.lowerBound]
        out += group(m, 1, in: s) == nil ? String(s[r]) : "z" + (group(m, 2, in: s) ?? "")
        last = r.upperBound
    }
    return out + s[last...]
}

/// z4z4z2 -> z8z2: merge runs of rests (between chord symbols, within a bar) into the native lengths.
private func mergeRests(_ s: String) -> String {
    let run = try! NSRegularExpression(pattern: #"(?:z[0-9]*)+"#)
    var out = "", last = s.startIndex
    for m in run.matches(in: s, range: NSRange(s.startIndex..., in: s)) {
        let r = Range(m.range, in: s)!
        out += s[last..<r.lowerBound]
        var total = s[r].split(separator: "z", omittingEmptySubsequences: false).dropFirst().reduce(0) { $0 + (Int($1) ?? 1) }
        for a in [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1] {
            while total >= a { out += a == 1 ? "z" : "z\(a)"; total -= a }
        }
        last = r.upperBound
    }
    return out + s[last...]
}
