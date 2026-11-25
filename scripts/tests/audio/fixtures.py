"""Generate the MP3 fixtures the audio regressions decode.

Both runners call this before compiling. The files are only regenerated when they are
missing or were produced by a different version of this script, so a CI cache can hold
them and ffmpeg is only needed on a cache miss.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/audio-tests"
STAMP = OUTPUT / ".fixtures-stamp"
FIXTURES = (
    "long.mp3",
    "short.mp3",
    "long-noxing.mp3",
    "long-44k-noxing.mp3",
    "long-vbr.mp3",
    "long-44k.mp3",
    "long-22k.mp3",
    "long-8k.mp3",
)


def version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def current():
    try:
        return STAMP.read_text().strip() == version() and all(
            (OUTPUT / name).is_file() for name in FIXTURES
        )
    except OSError:
        return False


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def lame_tag(data):
    at = 0
    if data[:3] == b"ID3":
        at = 10 + ((data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9])
    while at + 4 <= len(data) and (data[at] != 0xFF or data[at + 1] & 0xE0 != 0xE0):
        at += 1
    version = (data[at + 1] >> 3) & 3
    mono = data[at + 3] >> 6 == 3
    side = (17 if mono else 32) if version == 3 else (9 if mono else 17)
    tag = at + 4 + (0 if data[at + 1] & 1 else 2) + side
    magic = data[tag : tag + 4]
    if magic not in (b"Xing", b"Info"):
        return None, None
    flags = int.from_bytes(data[tag + 4 : tag + 8], "big")
    pos = tag + 8
    for bit, size in ((1, 4), (2, 4), (4, 100), (8, 4)):
        if flags & bit:
            pos += size
    if data[pos : pos + 4] in (b"LAME", b"Lavc", b"Lavf"):
        return magic.decode(), pos + 21
    return magic.decode(), None


def first_frame_tag(path):
    data = path.read_bytes()
    magic, at = lame_tag(data)
    return magic, int.from_bytes(data[at : at + 3], "big") >> 12 if at is not None else 0


# ffmpeg 9.1 declares 47 for MPEG-2 where 9.0 declares 576, so the low-rate fixtures pin it.
def declare_delay(path, frames):
    data = bytearray(path.read_bytes())
    _, at = lame_tag(data)
    data[at] = frames >> 4
    data[at + 1] = (frames & 0xF) << 4 | data[at + 1] & 0xF
    path.write_bytes(data)


def verify():
    expected = {
        "long.mp3": ("Info", 576),
        "long-vbr.mp3": ("Xing", 576),
        "long-44k.mp3": ("Info", 576),
        "long-22k.mp3": ("Info", 576),
        "long-8k.mp3": ("Info", 576),
        "long-noxing.mp3": (None, 0),
        "long-44k-noxing.mp3": (None, 0),
    }
    for name, want in expected.items():
        got = first_frame_tag(OUTPUT / name)
        if got != want:
            sys.exit(f"{name}: expected header {want}, got {got}")


def generate():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required to generate the MP3 regression fixtures")
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=240",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-b:a",
        "64k",
        str(OUTPUT / "long.mp3"),
    )
    ffmpeg("-i", str(OUTPUT / "long.mp3"), "-t", "4", "-c", "copy", str(OUTPUT / "short.mp3"))
    ffmpeg(
        "-i",
        str(OUTPUT / "long.mp3"),
        "-c",
        "copy",
        "-write_xing",
        "0",
        str(OUTPUT / "long-noxing.mp3"),
    )
    ffmpeg(
        "-i",
        str(OUTPUT / "long.mp3"),
        "-ar",
        "44100",
        "-b:a",
        "64k",
        "-write_xing",
        "0",
        str(OUTPUT / "long-44k-noxing.mp3"),
    )
    ffmpeg("-i", str(OUTPUT / "long.mp3"), "-q:a", "4", str(OUTPUT / "long-vbr.mp3"))
    ffmpeg(
        "-i", str(OUTPUT / "long.mp3"), "-ar", "44100", "-b:a", "64k", str(OUTPUT / "long-44k.mp3")
    )
    ffmpeg(
        "-i", str(OUTPUT / "long.mp3"), "-ar", "22050", "-b:a", "64k", str(OUTPUT / "long-22k.mp3")
    )
    ffmpeg(
        "-i", str(OUTPUT / "long.mp3"), "-ar", "8000", "-b:a", "64k", str(OUTPUT / "long-8k.mp3")
    )
    for name in ("long-22k.mp3", "long-8k.mp3"):
        declare_delay(OUTPUT / name, 576)
    STAMP.write_text(version() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even when the fixtures are current"
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.force or not current():
        generate()
    else:
        print(f"MP3 fixtures are current in {OUTPUT}")
    verify()
