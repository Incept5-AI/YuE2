import AVFoundation
import Foundation

/// Write interleaved stereo float samples as 16-bit PCM WAV (matches generate.py's write_wav).
public func writeWAV(_ samples: [Float], to url: URL, sampleRate: Int = sampleRate, channels: Int = 2) throws {
    var data = Data()
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let bytes = UInt32(samples.count * 2)
    data.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes)
    data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(UInt16(channels))
    u32(UInt32(sampleRate)); u32(UInt32(sampleRate * channels * 2)); u16(UInt16(channels * 2)); u16(16)
    data.append(contentsOf: Array("data".utf8)); u32(bytes)
    data.reserveCapacity(data.count + samples.count * 2)
    var pcm = [Int16](repeating: 0, count: samples.count)
    for (i, s) in samples.enumerated() { pcm[i] = Int16((max(-1, min(1, s)) * 32767).rounded()) }
    pcm.withUnsafeBytes { data.append(contentsOf: $0) }
    try data.write(to: url)
}

/// Encode to AAC in an .m4a container (256 kbps) for sharing.
public func writeM4A(_ samples: [Float], to url: URL, sampleRate: Int = sampleRate) throws {
    try? FileManager.default.removeItem(at: url)
    let settings: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: sampleRate,
                                   AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 256_000]
    let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let format = file.processingFormat
    let frames = samples.count / 2, block = 48_000
    var at = 0
    while at < frames {
        let n = min(block, frames - at)
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(n)) else { break }
        buf.frameLength = AVAudioFrameCount(n)
        let l = buf.floatChannelData![0], r = buf.floatChannelData![1]
        for i in 0..<n { l[i] = samples[2 * (at + i)]; r[i] = samples[2 * (at + i) + 1] }
        try file.write(from: buf)
        at += n
    }
}

/// Re-wrap an .m4a with iTunes metadata (title, artist, album, genre, cover art) so it looks right after
/// import into Music on a Mac (and, with Sync Library, on the phone). Passthrough: no re-encode.
public func tagM4A(_ url: URL, title: String, artist: String = "YuE2 Studio", album: String = "YuE2 Studio",
                   genre: String? = nil, comment: String? = nil, artwork: Data? = nil) async throws {
    func item(_ id: AVMetadataIdentifier, _ value: Any, _ type: String? = nil) -> AVMetadataItem {
        let m = AVMutableMetadataItem()
        m.identifier = id
        m.value = value as? NSCopying & NSObjectProtocol
        if let type { m.dataType = type }
        m.extendedLanguageTag = "und"
        return m
    }
    var items = [item(.iTunesMetadataSongName, title as NSString), item(.iTunesMetadataArtist, artist as NSString),
                 item(.iTunesMetadataAlbum, album as NSString), item(.iTunesMetadataAlbumArtist, artist as NSString)]
    if let genre { items.append(item(.iTunesMetadataUserGenre, genre as NSString)) }
    if let comment { items.append(item(.iTunesMetadataUserComment, comment as NSString)) }
    if let artwork { items.append(item(.iTunesMetadataCoverArt, artwork as NSData, kCMMetadataBaseDataType_PNG as String)) }

    let asset = AVURLAsset(url: url)
    guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
        throw NSError(domain: "YuE2Kit", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot create export session"])
    }
    export.metadata = items
    let tmp = url.deletingLastPathComponent().appending(path: "tagged-\(UUID().uuidString).m4a")
    try await export.export(to: tmp, as: .m4a)
    _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
}
