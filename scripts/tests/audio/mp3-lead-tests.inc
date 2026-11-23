static void mp3LeadTest() {
    struct Fixture {
        char const* path;
        ma_uint32 rate, lead;
        char const* label;
    };

    Fixture const fixtures[] = {
        {"artifacts/audio-tests/long.mp3",
         48000,
         2257,
         "an Info header delays the first audio by its frame and the encoder delay"},
        {"artifacts/audio-tests/long-vbr.mp3",
         48000,
         1105,
         "a Xing header delays the first audio by the encoder delay"},
        {"artifacts/audio-tests/long-44k.mp3",
         44100,
         2457,
         "a 44.1 kHz Info header scales the delay to 48 kHz"},
        {"artifacts/audio-tests/long-22k.mp3",
         22050,
         3659,
         "an MPEG-2 Info header counts its 576-sample frame"},
        {"artifacts/audio-tests/long-8k.mp3",
         8000,
         10086,
         "a lead longer than the decoder's window stays whole"},
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
                    ours.begin(),
                    ours.begin() + lead * 2,
                    [](float s) {
                        return s == 0;
                    }
                ) &&
                std::equal(theirs.begin(), theirs.end(), ours.begin() + lead * 2),
            label
        );

        if (lead) {
            auto playsFrom = [&](double seconds, double from, bool played) {
                Decoder fresh;
                ma_decoder freshReference{};
                if (!fresh.open(path) ||
                    ma_decoder_init_file(path, &config, &freshReference) != MA_SUCCESS)
                    return false;
                bool good = true;
                if (from >= 0) {
                    ours.assign(480 * 2, 0.f);
                    good = fresh.seek(from);
                    if (played) fresh.render(ours.data(), 480, 1.);
                }
                size_t const zeros = lead - size_t(seconds * 48000);
                good &= fresh.seek(seconds);
                ours.assign((zeros + 4800) * 2, -1.f);
                for (size_t at = 0; good && at < zeros + 4800; at += 480) {
                    size_t const count = std::min<size_t>(480, zeros + 4800 - at);
                    fresh.render(ours.data() + at * 2, count, 1.);
                    good &= !fresh.starvedBeforeEnd();
                }
                good &= reference(freshReference, 0, 4800) &&
                    std::all_of(
                            ours.begin(),
                            ours.begin() + zeros * 2,
                            [](float s) {
                                return s == 0;
                            }
                    ) &&
                    std::equal(theirs.begin(), theirs.end(), ours.begin() + zeros * 2);
                ma_decoder_uninit(&freshReference);
                return good;
            };
            bool inside = playsFrom(0, 10, false) && playsFrom(0.015625, -1, false) &&
                playsFrom(0.015625, 0, true);
            if (lead > 9000) inside &= playsFrom(0.1875, 0, true) && playsFrom(0.140625, 0, true);
            check(inside, (std::string(label) + ", a seek into the lead plays the rest of it").c_str());
        }

        // A fresh pair per position keeps the resampler's history identical on both sides.
        bool positioned = decoder.testLeadFrames() == lead;
        Decoder::testIndexFailsOnNextOpen = true;
        Decoder unindexed;
        positioned &= unindexed.open(path) && unindexed.testLeadFrames() == lead;
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
