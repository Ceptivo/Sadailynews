"""OPTIONAL. The voiceover is already in assets/vo_*.npy - only run this to regenerate it.
Needs models/kokoro-v1.0.onnx and models/voices-v1.0.bin from
https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
Voice af_heart, speed 1.08, hand-written misaki (US) phonemes.
Usage: python3 tts.py [segment ...]   (no args = all segments)"""
import json, sys
import numpy as np
import onnxruntime as rt


import os
DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB = json.load(open(f"{DIR}/assets/kokoro_vocab.json", encoding="utf-8"))["vocab"]
SR = 24000

SEGMENTS = [
    ("intro", "ðˈɪs ɪz ˈɛs ˈA dˈAli bɹˈif, jʊɹ sˈɪksti sˈɛkənd nˈuz ˈʌpdAt fɔɹ tˈuzdA, ðə θˈɜɹd ʌv nOvˈɛmbɚ."),
    ("vote", "spˈɛʃəl vˈOɾɪŋ ɹˈæps ˈʌp tədˈA, əhˈɛd ʌv təmˈɑɹOz lˈOkəl ɡˈʌvɚnmənt ɪlˈɛkʃənz. "
             "ðə ˈI ˈi sˈi sˈɛz mˈɔɹ ðən sˈɛvən hˈʌndɹəd θˈWzənd vˈOts wɚ kˈæst ɑn dˈA wˈʌn əlˈOn."),
    ("agoa", "wˈɑʃɪŋtən sˈɛz ɪt ɪntˈɛndz tə ɹəmˈuv sˈWθ ˈæfɹɪkə fɹʌm ɑɡˈOə, ðə dˈuɾi fɹˈi tɹˈAd dˈil, "
             "fɹʌm ʤˈænjuˌɛɹi fˈɜɹst. pɹətˈɔɹiə sˈɛz ɪt wɪl səpˈɔɹt əfˈɛktəd ˈɪndəstɹiz."),
    ("water", "ɪn ʧwˈɑnɛ, bɹˈɔŋkhɔɹstˌspɹAt ɹˈɛzɪdənts ɑɹ tˈOld ðɛɹ tˈæp wˈɔɾɚ ɪz stˈɪl nˈɑt sˈAf tə dɹˈɪŋk."),
    ("film", "fˈɪlm ˈɪndəstɹi tˈɔks wɪð ðə tɹˈAd dəpˈɑɹtmənt hæv bɹˈOkən dˈWn fɔɹ ə θˈɜɹd tˈIm."),
    ("rugby", "ɪn ɹˈʌɡbi, ɹˈɑsi ɪɹˈæzməs hæz nˈAmd sˈɪks ʌnkˈæpt plˈAɚz fɔɹ ðə spɹˈɪŋbɑks tˈʊɹ ʌv "
              "ˈɪɾəli, fɹˈæns, ænd ˈIɚlənd."),
    ("outro", "ðˈæts jʊɹ sˈɪksti sˈɛkənz. fˈɑlO fɔɹ lˈIv ɪlˈɛkʃən ˈʌpdAts təmˈɑɹO."),
]


def tokens(ph):
    missing = {c for c in ph if c not in VOCAB}
    if missing:
        sys.exit(f"not in vocab: {missing!r}")
    return [VOCAB[c] for c in ph]


def trim(a, thr=0.012):
    idx = np.where(np.abs(a) > thr)[0]
    return a[max(idx[0] - 600, 0): idx[-1] + 2400]


def main(only=None, voice="af_heart", speed=1.08):
    sess = rt.InferenceSession(f"{DIR}/models/kokoro-v1.0.onnx", providers=["CPUExecutionProvider"])
    names = {i.name: i for i in sess.get_inputs()}
    tok_name = "input_ids" if "input_ids" in names else "tokens"
    voices = np.load(f"{DIR}/models/voices-v1.0.bin")
    v = voices[voice]
    out = {}
    for key, ph in SEGMENTS:
        if only and key not in only:
            continue
        # split on sentence ends to keep each run short; join with a breath
        parts = [p.strip() for p in ph.replace(". ", ".|").split("|") if p.strip()]
        audio = []
        for p in parts:
            t = tokens(p)
            feed = {tok_name: np.array([[0, *t, 0]], dtype=np.int64),
                    "style": v[len(t) - 1].astype(np.float32),
                    "speed": np.array([speed], dtype=np.float32)}
            a = sess.run(None, feed)[0].ravel()
            audio += [trim(a), np.zeros(int(0.22 * SR), np.float32)]
        a = np.concatenate(audio[:-1])
        out[key] = a
        np.save(f"{DIR}/assets/vo_{key}.npy", a)
        print(key, round(len(a) / SR, 2))
    print("total", round(sum(len(a) for a in out.values()) / SR, 2))


if __name__ == "__main__":
    # python3 tts.py            -> all segments
    # python3 tts.py intro outro -> only those segments
    main(set(sys.argv[1:]) or None)
