// FMOD's MP3 playback keeps the encoder delay dr_mp3 trims and decodes an Info frame as one frame
// of silence. The plugin adds that time back so its song clock matches the game's reported position.
static void mp3LeadTest() {
    struct Fixture {
        char const* path;
        ma_uint32 rate, lead;
        char const* label;
    };
    Fixture const fixtures[] = {
        {"artifacts/audio-tests/long.mp3", 48000, 2257,
         "an Info header delays the first audio by its frame and the encoder delay"},
        {"artifacts/audio-tests/long-vbr.mp3", 48000, 1105,
         "a Xing header delays the first audio by the encoder delay"},
        {"artifacts/audio-tests/long-44k.mp3", 44100, 2457,
         "a 44.1 kHz Info header scales the delay to 48 kHz"},
        {"artifacts/audio-tests/long-noxing.mp3", 48000, 0, "an untagged MP3 keeps its timing"},
    };
    auto config = ma_decoder_config_init(ma_format_f32, 2, 48000);
    config.seekPointCount = 2048;
    std::vector<float> ours, theirs;
    auto reference = [&](ma_decoder& plain, ma_uint64 frame, size_t frames) {
        theirs.assign(frames * 2, 0.f);
        return ma_decoder_seek_to_pcm_frame(&plain, frame) == MA_SUCCESS &&
            ma_decoder_read_pcm_frames(&plain, theirs.data(), frames, nullptr) == MA_SUCCESS;
    };
    for (auto const& [path, rate, lead, label] : fixtures) {
        Decoder decoder;
        ma_decoder plain{};
        if (!decoder.open(path) || ma_decoder_init_file(path, &config, &plain) != MA_SUCCESS) {
            check(false, label);
            continue;
        }
        // The lead is song time before the file's first frame, followed by that frame without a gap.
        ours.assign((lead + 4800) * 2, -1.f);
        bool sought = decoder.seek(0), starved = false;
        for (size_t at = 0; sought && at < lead + 4800; at += 480) {
            size_t const count = std::min<size_t>(480, lead + 4800 - at);
            decoder.render(ours.data() + at * 2, count, 1.);
            starved |= decoder.starvedBeforeEnd();
        }
        check(
            sought && !starved && reference(plain, 0, 4800) &&
                std::all_of(
                    ours.begin(), ours.begin() + lead * 2, [](float s) {
                        return s == 0;
                    }
                ) &&
                std::equal(theirs.begin(), theirs.end(), ours.begin() + lead * 2),
            label
        );

        if (lead) {
            // A seek from outside the lead must pad it again on the backend path.
            Decoder again;
            ma_decoder againReference{};
            bool const opened = again.open(path) &&
                ma_decoder_init_file(path, &config, &againReference) == MA_SUCCESS;
            bool padded = opened;
            if (opened) {
                padded = again.seek(10) && again.seek(0);
                ours.assign((lead + 4800) * 2, -1.f);
                for (size_t at = 0; padded && at < lead + 4800; at += 480) {
                    size_t const count = std::min<size_t>(480, lead + 4800 - at);
                    again.render(ours.data() + at * 2, count, 1.);
                    padded &= !again.starvedBeforeEnd();
                }
                padded &= reference(againReference, 0, 4800) &&
                    std::all_of(
                        ours.begin(), ours.begin() + lead * 2, [](float s) {
                            return s == 0;
                        }
                    ) &&
                    std::equal(theirs.begin(), theirs.end(), ours.begin() + lead * 2);
                ma_decoder_uninit(&againReference);
            }
            check(padded, (std::string(label) + ", a later seek pads the lead again").c_str());
        }

        // A fresh pair per position keeps the resampler's history identical on both sides.
        bool positioned = decoder.testLeadFrames() == lead;
        for (double t : {1., 30.}) {
            Decoder fresh;
            ma_decoder freshReference{};
            bool const opened = fresh.open(path) &&
                ma_decoder_init_file(path, &config, &freshReference) == MA_SUCCESS;
            positioned &= opened;
            if (!opened) continue;
            ours.assign(4800 * 2, 0.f);
            positioned &= fresh.seek(t);
            fresh.render(ours.data(), 4800, 1.);
            positioned &= std::abs(fresh.position - (t + .1)) < 1e-9;
            positioned &= reference(freshReference, ma_uint64(t * 48000) - lead, 4800);
            positioned &= ours == theirs;
            ma_decoder_uninit(&freshReference);
        }
        if (rate == 48000) {
            // A seek inside the retained window must land on the same song time.
            Decoder window;
            ma_decoder windowReference{};
            bool const opened = window.open(path) &&
                ma_decoder_init_file(path, &config, &windowReference) == MA_SUCCESS;
            positioned &= opened;
            if (opened) {
                ours.assign(4800 * 2, -1.f);
                positioned &= window.seek(30);
                window.render(ours.data(), 4800, 1.);
                ours.assign(480 * 2, 0.f);
                positioned &= window.seek(30.0625);
                window.render(ours.data(), 480, 1.);
                positioned &= reference(windowReference, ma_uint64(30.0625 * 48000) - lead, 480);
                positioned &= ours == theirs;
                ma_decoder_uninit(&windowReference);
            }
        }
        check(positioned, (std::string(label) + ", at the game's song time").c_str());

        if (lead) {
            ma_uint64 const length = decoder.testLengthFrames();
            bool const guarded = decoder.hasAudioAt(double(length - 4000) / 48000.) &&
                !decoder.hasAudioAt(double(length + lead) / 48000.) &&
                decoder.seek((double(length) + lead / 2.) / 48000.) &&
                !decoder.seek((double(length + lead) + 480) / 48000.);
            check(guarded, (std::string(label) + ", the end guard includes the delay").c_str());
        }
        ma_decoder_uninit(&plain);
    }
}
