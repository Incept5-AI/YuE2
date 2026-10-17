import Foundation
import Observation
import os

/// Where the YuE2 weights live on the device, and the downloader that fetches them from Hugging Face.
/// Layout: Documents/Models/<variant>/{config.json, model.safetensors, …} — the same flat files as the
/// `ahmadw/YuE2-3B-MLX` snapshot's variant folder, so they can also be pushed over USB with devicectl.
@MainActor @Observable
final class ModelStore: NSObject {
    static let repo = "ahmadw/YuE2-3B-MLX"
    nonisolated static let files = ["config.json", "yue2_generation_config.json", "vae_config.json", "qwen.tiktoken",
                        "vae.safetensors", "model.safetensors"]
    nonisolated static let variants: [(id: String, label: String, gb: Double)] = [
        ("4bit", "4-bit (recommended)", 3.4), ("8bit", "8-bit (slower, more memory)", 4.2),
    ]

    var downloading: String?
    var downloadFraction = 0.0
    var downloadDetail = ""
    var lastError: String?
    private var task: URLSessionDownloadTask?
    private var continuation: CheckedContinuation<URL, Error>?
    private var session: URLSession!

    override init() {
        super.init()
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForResource = 6 * 3600
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: .main)
    }

    nonisolated static var root: URL {
        URL.documentsDirectory.appending(path: "Models", directoryHint: .isDirectory)
    }

    nonisolated static func directory(_ variant: String) -> URL { root.appending(path: variant, directoryHint: .isDirectory) }

    nonisolated static func isInstalled(_ variant: String) -> Bool {
        files.allSatisfy { FileManager.default.fileExists(atPath: directory(variant).appending(path: $0).path) }
    }

    nonisolated static var installed: [String] { variants.map(\.id).filter(isInstalled) }

    nonisolated static func sizeOnDisk(_ variant: String) -> Int64 {
        files.reduce(0) { sum, f in
            let a = try? FileManager.default.attributesOfItem(atPath: directory(variant).appending(path: f).path)
            return sum + ((a?[.size] as? NSNumber)?.int64Value ?? 0)
        }
    }

    func delete(_ variant: String) {
        try? FileManager.default.removeItem(at: Self.directory(variant))
    }

    func cancel() { task?.cancel() }

    func download(_ variant: String) async {
        guard downloading == nil else { return }
        downloading = variant; lastError = nil; downloadFraction = 0
        defer { downloading = nil; task = nil }
        let dir = Self.directory(variant)
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            var rv = URLResourceValues(); rv.isExcludedFromBackup = true
            var root = Self.root; try? root.setResourceValues(rv)
            for (i, f) in Self.files.enumerated() {
                let dest = dir.appending(path: f)
                if FileManager.default.fileExists(atPath: dest.path) { continue }
                downloadDetail = "\(f) (\(i + 1) of \(Self.files.count))"
                let url = URL(string: "https://huggingface.co/\(Self.repo)/resolve/main/\(variant)/\(f)")!
                let tmp = try await fetch(url)
                try? FileManager.default.removeItem(at: dest)
                try FileManager.default.moveItem(at: tmp, to: dest)
            }
            downloadDetail = "Done"
        } catch {
            lastError = (error as NSError).code == NSURLErrorCancelled ? "Download cancelled" : error.localizedDescription
        }
    }

    private func fetch(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { c in
            continuation = c
            task = session.downloadTask(with: url)
            task!.resume()
        }
    }
}

extension ModelStore: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The temp file is deleted when this returns, so move it somewhere stable first.
        let keep = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        let moved = Result { try FileManager.default.moveItem(at: location, to: keep) }
        MainActor.assumeIsolated {
            if status != 200 { continuation?.resume(throwing: PipelineErrorUI("HTTP \(status) from Hugging Face")) }
            else if case .failure(let e) = moved { continuation?.resume(throwing: e) }
            else { continuation?.resume(returning: keep) }
            continuation = nil
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                                totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        MainActor.assumeIsolated {
            if totalBytesExpectedToWrite > 0 { downloadFraction = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) }
            let name = downloadTask.originalRequest?.url?.lastPathComponent ?? ""
            downloadDetail = String(format: "%@ · %.2f of %.2f GB", name, Double(totalBytesWritten) / 1e9,
                                    Double(max(totalBytesExpectedToWrite, 0)) / 1e9)
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        MainActor.assumeIsolated {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}

struct PipelineErrorUI: LocalizedError {
    var message: String
    init(_ m: String) { message = m }
    var errorDescription: String? { message }
}

enum DeviceInfo {
    /// Memory iOS will still let this process allocate before jetsam (the real budget, not physical RAM).
    static var availableGB: Double { Double(os_proc_available_memory()) / 1e9 }
    static var physicalGB: Double { Double(ProcessInfo.processInfo.physicalMemory) / 1e9 }
    static var thermal: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}
