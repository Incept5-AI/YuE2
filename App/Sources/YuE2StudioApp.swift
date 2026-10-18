import SwiftUI
import YuE2Kit

@main
struct YuE2StudioApp: App {
    @State private var studio = Studio()
    @Environment(\.scenePhase) private var phase

    init() {
        // Keep MLX's buffer pool small: the phone's memory budget matters more than reuse speed.
        Memory.cacheLimit = 256 * 1024 * 1024
    }

    var body: some Scene {
        WindowGroup {
            RootView().environment(studio).task { studio.autorunIfRequested() }
        }
        .onChange(of: phase) { _, p in studio.setBackground(p == .background) }
    }
}

struct RootView: View {
    @State private var tab = 0

    var body: some View {
        TabView(selection: $tab) {
            CreateView().tabItem { Label("Create", systemImage: "music.mic") }.tag(0)
            LibraryView(tab: $tab).tabItem { Label("Library", systemImage: "music.note.list") }.tag(1)
            ModelView().tabItem { Label("Model", systemImage: "cpu") }.tag(2)
        }
    }
}

struct ModelView: View {
    @Environment(Studio.self) private var studio
    @State private var tick = 0

    var body: some View {
        @Bindable var s = studio
        NavigationStack {
            Form {
                Section {
                    ForEach(ModelStore.variants, id: \.id) { v in
                        let installed = ModelStore.isInstalled(v.id)
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(v.label).font(.headline)
                                Spacer()
                                if installed { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
                            }
                            Text(installed ? String(format: "Installed · %.2f GB", Double(ModelStore.sizeOnDisk(v.id)) / 1e9)
                                           : String(format: "%.1f GB download from Hugging Face", v.gb))
                                .font(.caption).foregroundStyle(.secondary)
                            if studio.store.downloading == v.id {
                                ProgressView(value: studio.store.downloadFraction)
                                Text(studio.store.downloadDetail).font(.caption).foregroundStyle(.secondary)
                                Button("Cancel download", role: .destructive) { studio.store.cancel() }.buttonStyle(.borderless)
                            } else if installed {
                                HStack {
                                    Button(studio.variant == v.id ? "In use" : "Use this") { s.variant = v.id }
                                        .disabled(studio.variant == v.id)
                                    Spacer()
                                    Button("Delete", role: .destructive) { studio.unloadModel(); studio.store.delete(v.id); tick += 1 }
                                }.buttonStyle(.borderless)
                            } else {
                                Button("Download") { Task { await studio.store.download(v.id); tick += 1 } }
                                    .buttonStyle(.borderless).disabled(studio.store.downloading != nil)
                            }
                        }
                        .id("\(v.id)-\(tick)")
                    }
                    if let e = studio.store.lastError { Text(e).foregroundStyle(.red).font(.footnote) }
                } header: { Text("YuE2-3B weights") } footer: {
                    Text("From ahmadw/YuE2-3B-MLX. Stay on Wi-Fi with the app open while downloading.")
                }

                Section("This device") {
                    TimelineView(.periodic(from: .now, by: 2)) { _ in
                        VStack(alignment: .leading, spacing: 4) {
                            LabeledContent("Model", value: Studio.machine)
                            LabeledContent("RAM", value: String(format: "%.1f GB", DeviceInfo.physicalGB))
                            LabeledContent("App memory budget left", value: String(format: "%.2f GB", DeviceInfo.availableGB))
                            LabeledContent("MLX active", value: String(format: "%.2f GB", Double(Memory.activeMemory) / 1e9))
                            LabeledContent("Thermal state", value: DeviceInfo.thermal)
                        }
                    }
                    Button("Unload model from memory") { studio.unloadModel() }.disabled(studio.running != nil)
                }

                Section("Writing assistant") {
                    LabeledContent("Model", value: Assistant.variantName)
                    LabeledContent("Status", value: Assistant.availability ?? "Ready, on device")
                }

                Section("About") {
                    Text("YuE2 (m-a-p) runs here through MLX on the GPU: score planning, music tokens, 32-step flow matching and the VAE decoder, ported from the Python MLX runner. Lyrics and style help come from Apple's built-in model.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Link("Open source from Incept5 (i5): github.com/Incept5-AI/YuE2",
                         destination: URL(string: "https://github.com/Incept5-AI/YuE2")!)
                        .font(.footnote)
                    Text("YuE2 weights: CC BY-NC 4.0, non-commercial use only.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Model")
        }
    }
}
