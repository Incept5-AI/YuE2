# CLAUDE.md

Guidance for Claude Code in this repo: YuE2-3B song generation running natively on iPhone and iPad (Swift + MLX).
This is an [Incept5 (i5)](https://incept5.ai) open-source project: `github.com/Incept5-AI/YuE2`, one
repo in the [Incept5-AI](https://github.com/Incept5-AI) collection (the org profile lists the others). `README.md` has what the
app does, the measurements and the musical findings. This file covers building, testing and the traps.

## First run for a new user (do these in order)

1. **Check the tools.** Run `xcodebuild -version` (needs Xcode 27; the app targets iOS 27) and `xcodegen --version`
   (`brew install xcodegen` if it is missing).
2. **Set up signing.** If `Config/Local.xcconfig` is missing, copy `Config/Local.xcconfig.example` and ask the
   user for two values:
   - **Team ID:** Xcode → Settings → Accounts, or developer.apple.com → Membership.
   - **Bundle id prefix:** must be unique to them. **Never commit `Local.xcconfig`.**
3. **Check the account.** It must be a paid team signed in to Xcode (Settings → Accounts). A free team cannot
   grant `increased-memory-limit` (below).
4. **Generate the project.** Run `xcodegen generate`. Re-run it after any change to `project.yml` or after adding
   or removing source files: `YuE2Studio.xcodeproj` is generated, and committed only as a convenience.
5. **Build and install.** Use the commands below. Find the UDID with `xcrun devicectl list devices`.
6. **Get the weights onto the device.** Use the Model tab download (slow), or push them over USB (fast, below).

## Build, install, test

```bash
# App (signing runs outside the Claude sandbox: it needs the login keychain)
xcodegen generate
xcodebuild -project YuE2Studio.xcodeproj -scheme YuE2Studio -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> build/Build/Products/Release-iphoneos/YuE2Studio.app

# Compile check without any signing (CI, or before Local.xcconfig exists)
xcodebuild ... CODE_SIGNING_ALLOWED=NO build

# Mac CLI. Use xcodebuild, not `swift build`: only xcodebuild compiles MLX's Metal library.
cd YuE2Kit && xcodebuild -scheme yue2cli -destination 'platform=macOS' -derivedDataPath build -configuration Release build
YuE2Kit/build/Build/Products/Release/yue2cli --model <dir with the 6 files> --style "..." --lyrics-file L.txt \
  --seed 19590 --sem-max 750 --out test.wav         # 750 music tokens = 30 s; ~26 s on an M5 Max, peak 6.7 GB
```

**Weights.** `hf download ahmadw/YuE2-3B-MLX --include "4bit/*"` puts them in the Hugging Face cache.
`yue2cli --model` can point straight at `~/.cache/huggingface/hub/models--ahmadw--YuE2-3B-MLX/snapshots/<rev>/4bit`.
To push them to a device, copy them out with `cp -L` first: the cache holds symlinks. Then:

```bash
xcrun devicectl device copy to --device <UDID> --domain-type appDataContainer \
  --domain-identifier <BUNDLE_ID_PREFIX>.yue2studio --source <dir with the 6 files> --destination Documents/Models/4bit
```

The six files are `config.json`, `yue2_generation_config.json`, `vae_config.json`, `qwen.tiktoken`,
`vae.safetensors` and `model.safetensors`. The device needs **8bit** only for iPads.

**Test on a device without touching it:**

```bash
xcrun devicectl device process launch --device <UDID> --console \
  --environment-variables '{"YUE2_AUTORUN":"buble","YUE2_SECONDS":"60"}' <BUNDLE_ID_PREFIX>.yue2studio
```

- The app renders that preset, prints `YUE2 …` progress lines, and ends with `YUE2 DONE {json}` (timings, peak
  memory). `YUE2_STEPS`, `YUE2_VARIANT` and `YUE2_NAME` are also read.
- **The device must be unlocked**, or the launch is refused with "Locked". Retry in a loop until it succeeds.
- Copy the result back with `devicectl device copy from … --source Documents/Songs/<name>`.

**Parity check for instrumentals** (after any change to `Instrumental.swift`):

```bash
ls some/plans/*.abc > list.txt
yue2cli --silence-abc list.txt                    # writes <file>.swift-silenced
python3 -I tools/silence_vocal.py list.txt        # writes <file>.py-silenced, compares, exits 1 on any difference
```

The two were identical on 33 plans (521 groups changed) on 2026-10-17. To get plans, use the app's *Plan score*
or `tools/reference.py`. `tools/requirements.txt` lists what the Python tools need: a venv with mlx 0.32.3,
numpy, tiktoken and huggingface_hub. Never install into a global env.

## Architecture in one screen

The pipeline is `Pipeline.render`. It runs:

1. **Plan.** The ABC score is sampled by the language model (skipped when planning is off; replaced when the user
   supplies a score). For an instrumental style, `silenceVocal` runs on the plan.
2. **Music tokens.** These are semantic codec tokens, sampled with CFG, at about 25 per second of audio. The
   maximum is 9000.
3. **Flow matching.** 32 midpoint steps by default, in chunks.
4. **VAE decode.** 128-frame tiles, then 48 kHz stereo.

Stages 1–3 run inside a closure, so the model can be freed before decode (see the gotcha below).

The app's `Studio` owns UI state and one job at a time. `Worker` owns the `Yue2Pipeline` and refuses to load when
`os_proc_available_memory()` is below the weights plus 1 GB. `Gate` pauses GPU work while the app is in the
background and carries the cancel flag. Songs are saved to `Documents/Songs/<name>/` (song.m4a, song.wav,
score.abc, request.json, result.json).

## Gotchas (each cost a wrong turn)

- **Memory entitlement.** Without `com.apple.developer.kernel.increased-memory-limit`, iOS gives the app 3.53 GB
  on a 12 GB iPhone 18 Pro, and SIGKILLs it while it loads 3.64 GB of weights. There is no crash report; the
  device log says only "App terminated due to signal 9".
  - With the entitlement, the iPhone allows 6.43 GB and an M5 iPad Pro 17.2 GB.
  - `App/YuE2Studio.entitlements` also has `extended-virtual-addressing`.
  - Automatic signing needs `-allowProvisioningUpdates` and a signed-in paid account to mint a profile with
    them. A wildcard profile will not do.
- **XcodeGen silently ignores a target-level `resources:` key.** The first build shipped without
  `presets.json`, so the app opened with blank fields and an empty preset menu. Resources go under `sources:`
  with `buildPhase: resources`, as in `project.yml`.
- **Do not set `DEVELOPMENT_TEAM` or `PRODUCT_BUNDLE_IDENTIFIER` in `project.yml`.** Target settings there
  override the xcconfig, and would put someone's team ID into git. Choosing a team in Xcode's Signing pane
  writes it into `project.pbxproj`, which is just as bad; set it in `Config/Local.xcconfig` instead.
- **`devicectl device install` kills the running app** (signal 9, no crash report). An install that lands while
  someone is playing a song looks exactly like a playback crash. Don't install while the app is in use.
- **`devicectl device install` can fail with error 4016 ("unavailable")** when an iPhone is asleep or on flaky
  Wi-Fi. Wake it, plug it in, and retry; it succeeded on retry.
- **Signing from a Claude Code session must run unsandboxed** (`dangerouslyDisableSandbox`); the sandbox hides the
  login keychain, so code signing cannot find the identity.
- **Low-memory mode** switches on below a 9 GB budget (phones). It:
  - quantizes the 756 MB embedding table to 4-bit;
  - preallocates the music-token KV caches;
  - frees the language model before decode.

  It cut a full song's peak from 6.8 GB to 4.99 GB. iPads skip it and keep full quality.
- **A `render()` local holding the model defeats unloading.** `self.model = nil` freed nothing while
  `let model = try ensureModel()` was still in scope. That is why stages 1–3 run in a closure.
- **The sampler was 3.7× slower than Python** (65 vs 240 tok/s): `MLXArray(0..<VOCAB)` rebuilt a 184,704-entry
  array on the CPU every token. Masks are now built once per generation with `MLXArray.arange`, giving 224 tok/s.
  Look for per-token CPU work first if speed regresses.
- **VAE tile size:** a 128-frame core is bit-identical to the original 1024 and peaks at 3.4 GB instead of
  5.9 GB. A 64-frame core is *not* exact (max diff 3e-4).
- **iOS stops GPU work in the background.** Submitting a Metal command buffer from the background gets the app
  killed. `Gate.checkpoint()` runs between evaluations (per token, flow step and decode tile). The idle timer is
  disabled during jobs.
- **Parity with Python is close, not bit-exact.**
  - What matches: the tokenizer, on every preset and on a 2,252-token score; and layer-1 hidden states.
  - What drifts: after layer 1, bf16 rounding differences accumulate (logit gap about 0.03 on a scale of
    about 20). mlx-swift 0.32.3 bundles MLX core 0.32.2, while Python has 0.32.3.
  - So the same seed gives the same first ~10 tokens, then a different but equally valid song.
  - It is **not** `silu` (`x * sigmoid(x)` changed nothing).
  - Judge the port on quality instead: the Swift Celine take scored 0.0% lyric error and 0.944 melody match.
- **Python's `open().read()` turns `\r\n` into `\n`.** A tokenizer "mismatch" on CRLF text was the test harness.
- **`Memory.peakMemory = 0` resets the peak**; the setter ignores the value.
- **Foundation Models** is unavailable when Apple Intelligence is off, still downloading, or unsupported. In
  that case `Assistant.availability` returns the reason and the writing buttons disable themselves. The rules
  text in `Assistant.swift` is measured YuE2 behaviour, so keep it in step with README "What we learned".
- **Same seed, different runner, different song.** Preset seeds were chosen on the Python MLX runner on a Mac. On
  the device they are simply good starting points. Instrumental seeds especially are luck; see README
  "Instrumentals".

## Model musical gotchas (affect presets and any rules text you write)

- The ABC dialect is strict. Header lines are exactly `X:1`, `T:`, `M:`, `L:1/<pow2>`, `Q:1/4=<int>`, the two
  fixed `V:` definitions, then `K:`. After that:
  - Music comes in groups of 1–4 bars: a `V: Vocal` line, then a `V: Ins` line with the same bar count.
  - Chord symbols go in the Vocal line only. Spell suspended dominants `Ab7sus4`, not `Ab7sus`.
  - A key change is a `K:` line directly after **both** `V:` lines of the group where it starts.
  - Section comments are `% verse`, `% chorus` and so on.
  - `tools/vendor/abc_tools.py` validates a score: `python3 -I tools/vendor/abc_tools.py inspect score.abc`.
- The vocal line is written **an octave above** sung pitch.
- The prompt's BPM is ignored when the model plans; supply a score to fix the tempo.
- Songs run about 25% longer than the score. At 9000 tokens (6:00) the song is cut mid-line.
- Keep `female vocal` / `male vocal` literal. Vocal-technique words, including "no vibrato", do nothing.
- For instrumentals, use an `instrumental` tag, empty sections and planning Full. Don't add "no vocals".

## Conventions

- Keep the Swift and Python `silenceVocal` byte-identical. Change both, then run the parity check.
- Numbers in README are n=1 unless stated otherwise; say so when you add one. Date the findings you add.
- Never commit weights, renders (`*.wav`, `*.m4a`), `build/`, `Config/Local.xcconfig`, a team ID, or device UDIDs.
- GitHub Pages serves this repo as-is (`.nojekyll`, `main`, root) at https://incept5-ai.github.io/YuE2/. GitHub's own
  file view forces a download for audio, so link samples to the Pages URL so they play in the browser.
