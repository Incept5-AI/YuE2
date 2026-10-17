# YuE2 Studio: full songs generated on an iPhone or iPad

*An open-source project from [Incept5 (i5)](https://incept5.ai), part of [Incept5/incept5-ai](https://github.com/Incept5/incept5-ai).*

![YuE2 Studio icon](App/Icon/logo-512.png)

[YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) (m-a-p, 2026) writes a complete song from a style prompt and
lyrics: lead vocal and accompaniment, 48 kHz stereo, up to 6 minutes. This repo runs it **natively on the device**
with [mlx-swift](https://github.com/ml-explore/mlx-swift). There is no server, and after the one-time model
download nothing touches the network.

It has two parts:
- **YuE2Kit**, a Swift package that ports the
  [ahmadw/YuE2-3B-MLX](https://huggingface.co/ahmadw/YuE2-3B-MLX) Python runner to Swift. It covers the tokenizer,
  score planning, music tokens, flow matching and the VAE decoder.
- **YuE2 Studio**, a SwiftUI app for iPhone and iPad. Writing help (style tags, lyrics, revisions, questions)
  comes from Apple's built-in on-device model through the Foundation Models framework.

YuE2 differs from Suno-style models in one useful way: it first writes an **ABC music score** (melody, chords and
sections) and then sings that score. You can read the score, edit it, or write your own, and the model renders
what you give it.

> **Licence:** the YuE2 weights and the MLX runner this ports are **CC BY-NC 4.0, for non-commercial use only**,
> and so is this code (see [LICENSE](LICENSE)). The songs you make are subject to the model's licence.

## Quick start

**You need:**
- A Mac with **Xcode 27** and [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).
- A **paid Apple Developer account**. A free personal team cannot grant the increased-memory entitlement, and
  without it the app is killed while it loads the model (see "Memory").
- One of these devices on **iOS/iPadOS 27**:

  | Device | Status |
  |---|---|
  | iPad Pro M5 | Tested |
  | iPhone 18 Pro | Tested |
  | iPhone 17 Pro | Expected to work |
  | Other devices with at least 12 GB of RAM | Expected to work |
  | Devices with 8 GB of RAM | Untested, and unlikely to fit |

- Apple Intelligence switched on, but only for the writing-help buttons. Song generation works without it.

**Steps:**

```bash
git clone https://github.com/Incept5/incept5-ai.git && cd incept5-ai/YuE2
cp Config/Local.xcconfig.example Config/Local.xcconfig   # set DEVELOPMENT_TEAM and a unique BUNDLE_ID_PREFIX
xcodegen generate                                        # (re)creates YuE2Studio.xcodeproj from project.yml
open YuE2Studio.xcodeproj                                # select your device, then Run (Release is faster)
```

Then, in the app:
1. On the **Model** tab, download the 4-bit weights (3.4 GB from Hugging Face). Stay on Wi-Fi with the app open.
2. On the **Create** tab, press **Generate song**. The Michael Bublé-style swing preset is loaded, so the first
   press makes a song.

Running Claude Code in this folder? `CLAUDE.md` holds the build, deploy and test commands, and the traps.

## What the app does

- **17 presets** under "Start from a singer or instrumental". All lyrics are originals; the tags describe a
  sound, not an artist.
  - **10 singer styles:** Celine-style power ballad, Adele, Ed Sheeran, Michael Bublé, Ella Fitzgerald, 90s rap,
    Queen, Édith Piaf (French), Helene Fischer (German), and the model card's Mandarin city-pop demo.
  - **7 instrumentals for videos:** lo-fi study beats, cinematic trailer, acoustic vlog, corporate, synthwave,
    tropical house and ambient piano.
- **Every runner control.** The app exposes:
  - style and lyrics;
  - planning off / melody / full, with an editable ABC score and a *Plan score* button;
  - seed (with dice), max length (20–360 s), CFG (auto or manual) and flow steps (8–64);
  - temperature, top-p, top-k and repetition penalty, for both the score stage and the music stage.
- **On-device writing help** (Apple Foundation Models): *Write style*, *Write lyrics* (choose a language and
  song structure), *Revise* and *Ask*. Each sends YuE2's measured rules with the prompt (see "What we learned").
- **Library.** Play songs, or share them as m4a, lossless WAV or the `.abc` score. From any song you can use
  *Load these settings*, *Same song, new seed* or *Use this score*. Songs are also visible in the Files app under
  On My iPhone → YuE2 Studio → Songs.
- **Model tab.** Download or delete the 4-bit and 8-bit weights, and watch the live memory budget, MLX memory and
  thermal state.
- **Ready for Music.app.** iOS has no API for an app to add its own files to the Music library. So every m4a
  carries a title, artist, album ("YuE2 Studio"), genre and cover art. AirDrop it to a Mac and drop it into
  Music; with Sync Library on, it then shows up on the phone.

## Measurements (n=1 each, 2026-10)

| | Mac M5 Max (`yue2cli`) | iPad Pro M5 (16 GB) | iPhone 18 Pro (12 GB) |
|---|---|---|---|
| Memory iOS allows the app | n/a | 5.36 GB without the entitlement, **17.2 GB with it** | 3.53 GB without, **6.43 GB with** |
| Full Bublé song | 4-bit low-memory: 3:22 of audio in 1:36, peak 4.99 GB | **8-bit: 3:17 in 7:07** (plan 37 s, music 113 s, flow 262 s, decode 16 s), peak 7.1 GB | **4-bit low-memory: 3:24 in 9:14** (plan 35 s, music 115 s, flow 379 s, decode 25 s), peak 4.8 GB, thermal state "fair" at the end |
| 60 s Bublé, 4-bit | 12 s | 1:48 | |
| Lyric word error / melody match / chord match | Celine preset: 0.0% / 0.944 / 0.927 (Python 8-bit: 0.7% / 0.899 / 0.936) | 0.8% / 0.937 / 0.685 | 1.2% / 0.768 / 0.847 |
| Speed of music-token generation | 224 tok/s (Python MLX: 240) | | |

How these were scored:
- On a Mac, using Qwen3-ASR for lyrics and SheetSage2 for melody and chords. The melody figure is compared with
  the model's own plan; chance level is about 0.30.
- Each device sampled its own score, so the iPad and iPhone columns are two different songs. They are not a
  4-bit vs 8-bit comparison.
- **A phone takes about 2.7× real time** for a full song, and **an M5 iPad Pro about 2.2×**. Keep the app in
  the foreground: iOS does not allow GPU work in the background. The app pauses when backgrounded and resumes
  when you return.

## What we learned

### About the model (these apply on every platform)

- **The score is the control surface; the prompt is a weak lever.**
  - A hand-written score was sung at 0% word error and 0.97 melody match.
  - An edit to the chords left the melody untouched.
  - Prompt wording barely moved anything.
- **Style prompts work best as short comma-separated tags**, in the form
  *language, genre, female/male vocal, 3–5 instruments, BPM, 1–3 moods*.
  - Always include the literal tag `female vocal` or `male vocal`. With "pop diva" alone, 3 of 3 takes came out
    as male voices.
  - Naming acoustic guitars or a gospel choir pulls the song toward country.
- **These prompt ideas were tried and do not work:**
  - **Describing vocal technique** (vibrato, belting, twang, breathy). Four phrasings all kept about 90 cents of
    vibrato on long notes.
  - **"no vibrato".** It gave 92.7 cents against 93.2 without the tag.
  - **The prompt's BPM, when the model plans.** It wrote 138 BPM for a "68 BPM" request. To fix the tempo, supply
    a score (`Q:` line).
- **The score's vocal line is written an octave above the sung pitch.** Every plan we measured had a median of
  F5 and was sung around F4. If you write a score by hand, write the melody an octave up.
- **Renders run about 25% longer than the score's nominal length**, often with an extra chorus. The hard limit is
  9000 music tokens (6:00), where a song is cut mid-line, so keep scores under about 4:30.
- **Planning modes.**
  - **Full** (melody + chords) gave 0–3% lyric word error, against 14% with planning off.
  - **Melody** is the model card's recipe for covers; it keeps your melody and drops the chords.
- **Languages.** French (11% word error) and German (3.5%) work, although the card lists only English and
  Mandarin.
- **Weights.** 4-bit was about 40% faster than 8-bit with similar scores; bf16 scored no better than 8-bit.
  The 4-bit variant keeps the flow-matching path at 8 bits, because fully 4-bit flow matching drifts.
- **Seeds do not carry across runners.** MLX random numbers differ from PyTorch, so the model card's seeds do
  not reproduce the card's songs. Swift and Python MLX also diverge after about 10 tokens (see `CLAUDE.md`).
  Treat a seed as a re-roll button, not a recipe.

### Instrumentals

YuE2 has no instrumental switch. This recipe works:
- an `instrumental` tag in the style;
- lyrics made only of empty section tags (`[Intro]`, `[Interlude]`, `[Solo]`, `[Outro]`);
- **planning on Full**. With planning off, the model sang gibberish (vocal stem within 4–9 dB of the mix).

It has two failure modes:
- **On some seeds the planner writes a vocal melody anyway** (3 of 7 first takes), and the model then sings it.
  `silenceVocal` (`YuE2Kit/Sources/YuE2Kit/Instrumental.swift`) runs on every instrumental plan. It turns vocal
  notes into rests and moves the tune to the instrument line.
- **Wordless oohs and hums are seed luck.**
  - Leaving out the tag `no vocals` and naming sections Interlude/Solo rather than Verse/Chorus was clean on 3
    of 4 takes, against 1 of 4 (weak evidence). Like "no vibrato", naming the thing seems to invite it.
  - Even so, only 7 of 21 screened takes had no vocal at all. If a take hums, roll another seed.
  - A Mac can remove vocals afterwards with a stem separator (we used Mel-RoFormer); the phone cannot.

### About running it on iOS

- **Memory, not speed, is the limit.** Without the `increased-memory-limit` entitlement, iOS allowed only
  3.53 GB on a 12 GB iPhone, and the app was SIGKILLed while loading 3.64 GB of weights. The only log line was
  "App terminated due to signal 9".
- **Low-memory mode** switches on automatically when the budget is under 9 GB, which means phones. It cut a full
  song's peak from about 6.8 GB to 4.99 GB:
  - the 756 MB embedding table is quantized to 4-bit;
  - the music-token caches are allocated once rather than grown;
  - the language model is freed before the VAE decode.
- **VAE decode in 128-frame tiles** is bit-identical to the shipped 1024 and peaks at 3.4 GB instead of 5.9 GB.
  64-frame tiles are *not* exact.
- **The Swift port's sampler was first 3.7× slower than Python** (65 vs 240 tok/s), because it rebuilt a
  184,704-entry index array on the CPU every token. Build masks once per generation.
- **Ideas we considered and did not use:**
  - **mlx-swift-lm.** YuE2Kit depends only on `mlx-swift`. The model is not a standard LM (it mixes two
    transformers and a flow-matching head), and mlx-swift-lm's traits interact badly with Foundation Models.
  - **Running stem separation on the device** to clean instrumentals. It would add another model and its memory
    on top of a 4.8 GB peak.
  - **A cloud LLM for writing help.** Apple's on-device model is good enough for tags and lyrics once given the
    rules, and keeps everything on the device.

## Tips

- **Faster:** use 4-bit. Flow matching is the slowest stage on a device (about 70% of a phone render), and its
  time scales with the flow steps setting. We have not measured how much quality 16 steps costs against the
  default 32, so listen before relying on it. Try a 60 s max length before committing to a full song. Keep the
  device cool and plugged in; a hot phone throttles.
- **Better:** plan the score first, read it, and fix what you don't like (chords, key change, tempo) before
  generating. Keep lyric sections to 4–8 short lines with steady syllable counts, and repeat chorus text word
  for word. If a take is off (straight 8ths in a swing song, the wrong voice), re-roll the seed rather than
  rewording the prompt.
- **Getting weights on quickly:** the Model tab downloads from Hugging Face. During development, pushing them over
  USB with `devicectl` is much faster (see `CLAUDE.md`).

## Layout

| Path | What |
|---|---|
| `YuE2Kit/Sources/YuE2Kit/` | The port: `Tokenizer.swift` (tiktoken BPE), `Model.swift`, `VAE.swift`, `Pipeline.swift`, `Audio.swift` (WAV/m4a, tagging), `Instrumental.swift` |
| `YuE2Kit/Sources/yue2cli/` | Mac CLI for renders, parity checks and memory measurement |
| `App/Sources/` | SwiftUI app: `Studio` (state and jobs), `CreateView`, `AISheet` and `Assistant` (Apple's model), `LibraryView`, `ModelStore` (download) |
| `App/Resources/presets.json` | The 17 presets (style, lyrics, seed, settings) |
| `App/Icon/logo.svg` | Source of the icon |
| `Config/` | Signing: `Signing.xcconfig` (committed defaults) and your git-ignored `Local.xcconfig` |
| `project.yml` | XcodeGen spec; `YuE2Studio.xcodeproj` is generated from it |
| `tools/` | Optional Python references for checking the port on a Mac (`reference.py`, `silence_vocal.py`) |

## Credits

- [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) and YuE2-Vae: the model (CC BY-NC 4.0).
- [ahmadw/YuE2-3B-MLX](https://huggingface.co/ahmadw/YuE2-3B-MLX): the MLX weights and Python runner this
  ports (CC BY-NC 4.0).
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) (MIT).
- `tools/vendor/abc_tools.py`: the ABC validator from the upstream yue2-music skill (Apache-2.0).
- Port, app, measurements and presets: [Incept5](https://incept5.ai).
