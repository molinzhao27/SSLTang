"""PROTOCOL_V4 P-AC: per-TR low-level acoustic and timing features for a story.

Columns (``FEATURE_NAMES``, F = 37): a 32-band log-mel envelope, RMS, F0, word rate,
phoneme rate and word-onset count. TR t covers the half-open stimulus interval
[c_t - TR/2, c_t + TR/2) with c_t = tr_start + t * TR, from ``tr_centers`` exactly as the
semantic targets (``downsample_words``), so the acoustic row at TR t pairs with the
same design columns (t + 1..t + 3) as the 768-d target in W1.

- Audio is resampled to 16 kHz (``resample_poly``). Mel: 25-ms Hann frames, 10-ms hop,
  512-point FFT, 32 HTK-mel triangles over 0-8 kHz; log(power + floor) per frame with the
  floor 80 dB below the story's peak band power, then the mean over frames centred in the
  TR (no frame: the log floor). RMS: over the samples in the TR (none: 0). Every column is
  invariant to the audio gain after the story z-score.
- F0: median over voiced frames centred in the TR, 0 if none. The registered method is
  ``F0_METHOD = 'autocorr'``, a deterministic autocorrelation tracker (Boersma 1993, Praat
  defaults: 3-period Hann frames, window-corrected autocorrelation, local-maximum candidates
  ranked on their interpolated peak strength (1/4-sample band-limited lags, parabolic vertex),
  voicing 0.45, silence 0.03 of the story peak, octave cost 0.01), range 65-500 Hz, so the
  features never depend on which libraries happen to be installed. ``librosa.pyin`` runs
  only when requested by name. The method and library versions are reported in ``info``.
- Timing (TextGrid word and phone tiers; words through ``read_word_intervals``, so the
  BAD_WORDS filter matches the semantic targets): word rate and phoneme rate are words
  and ARPAbet phones (39 phonemes, stress removed) with midpoints in the TR per second;
  the onset count is words starting in the TR. Phones come from the story TextGrid's phone
  tier by default, or from a separate TextGrid or interval list (``phones=``). With the
  explicit ``phones=None`` (LPP: the official TextGrids have one word tier, 'TokensAlign')
  the phoneme-rate column is dropped (F = 36) and ``info['phone_source'] == 'none'``; that
  reduced set needs a registered deviation before any P-AC result is read. A phone source with
  no ARPAbet phone, over 5% non-ARPAbet speech labels (IPA), phones past the audio, or no phone
  in any TR raises ValueError instead of yielding a silent all-zero column.
- ``acoustic_targets`` z-scores every column over the story's TRs (std <= 1e-6: zero).
- ``story_features(story, run, manifest_path)`` is the runner entry point: it resolves the
  story's WAV (``<audio_root>/<story>.wav``, ``run['audio_path']``, or ds003020's
  ``stimuli/<story>.wav`` beside ``derivatives/TextGrids``) and TextGrid from the manifest.

Reading a story's audio or transcript is a target stimulus read (rule R1): the caller
records it with the provenance log and passes budget stories only. P-AC scores are the
nested out-of-fold r of ``encoding_scores`` with these features as predictors, exactly
as W1 with the semantic targets; weights follow the W1 soft rule (``column_weights``).
Rows outside the fold's fitting part are never read by ``encoding_scores``, so
``acoustic_rows(..., rows=fit_pos)`` fills them with zeros instead of reading their audio.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from math import gcd
from pathlib import Path
import numpy as np
from ..encoding_weights import encoding_scores
from ..semantic.features import BAD_WORDS, read_word_intervals, tr_centers

SR = 16000
N_MELS, N_FFT, WIN, HOP = 32, 512, 400, 160
MEL_FMIN, MEL_FMAX = 0.0, 8000.0
LOG_FLOOR = 1e-8  # Relative to the story's peak mel band power (-80 dB).
F0_MIN, F0_MAX = 65.0, 500.0
VOICING, SILENCE, OCTAVE_COST = 0.45, 0.03, 0.01
F0_UPSAMPLE = 4  # Autocorrelation lag resolution: 1/4 sample.
F0_METHOD = 'autocorr'  # Registered; 'pyin' only by name.
PYIN_FRAME = 1024
STD_FLOOR = 1e-6
CHUNK = 4096
END_TOLERANCE = 0.5  # Seconds a transcript may run past the end of the audio.
PHONE_DROP_MAX = 0.05  # Largest share of speech phone labels outside ARPAbet (ds003020 stories: <= 0.13%).
NON_SPEECH = frozenset(BAD_WORDS | {'sil', 'spn', 'cg'})  # Silence/noise phone labels (braces, case ignored).
FEATURE_NAMES = tuple(f'mel_{i:02d}' for i in range(N_MELS)) + (
    'rms', 'f0', 'word_rate', 'phoneme_rate', 'word_onset_count')
PHONEMES = frozenset('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH '
                     'UH UW V W Y Z ZH'.split())


def load_audio(source):
    """Mono float64 samples and rate from a WAV path or ``(samples, rate)``; integer PCM scaled to [-1, 1)."""
    if isinstance(source, (str, Path)):
        from scipy.io import wavfile
        rate, data = wavfile.read(str(source))
    else:
        data, rate = source
    data = np.asarray(data)
    if data.dtype == np.uint8:
        x = (data.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.integer):
        x = data.astype(np.float64) / float(-np.iinfo(data.dtype).min)
    else:
        x = data.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(1)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Audio must be a finite, non-empty mono or multi-channel signal')
    if int(rate) != rate or rate <= 0:
        raise ValueError('Audio sample rate must be a positive integer')
    return x, int(rate)


def resample(x, rate, target=SR):
    """Polyphase resampling to ``target`` Hz (identity when the rates match)."""
    if rate == target:
        return x
    g = gcd(int(rate), int(target))
    from scipy.signal import resample_poly
    return resample_poly(x, target // g, rate // g)


def _tiers(path, reference_dir=None):
    """Tier name -> (start, end, label) with the checked-in Tang/Huth parser (as ``read_word_intervals``)."""
    root = Path(reference_dir) if reference_dir else Path(__file__).resolve().parents[2] / 'reference/semantic-decoding'
    parser = root / 'decoding/utils_ridge/textgrid.py'
    if not parser.is_file():
        raise FileNotFoundError(f'Tang/Huth TextGrid parser not found: {parser}')
    spec = importlib.util.spec_from_file_location('_semantic_textgrid', parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = Path(path).read_bytes()  # The official LPP TextGrids are UTF-16 with a BOM.
    grid = module.TextGrid(raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8'))
    return {t.tier_name(): [(float(a), float(b), str(w).strip()) for a, b, w in t.make_simple_transcript()]
            for t in grid.tiers}


def _bare(label):
    return label.strip().strip('{}').strip()


def _is_word(label):
    return _bare(label).lower() not in BAD_WORDS


def _is_phone(label):
    return _bare(label).upper().rstrip('012') in PHONEMES


def _phone_tier(path, reference_dir=None):
    tiers = [v for k, v in _tiers(path, reference_dir).items() if 'phone' in k.lower()]
    if len(tiers) != 1:
        raise ValueError(f'{path}: expected exactly one explicitly named phone tier (or pass phones=None)')
    return tiers[0]


def read_timing(textgrid, reference_dir=None, phones='textgrid'):
    """``{'words', 'phones', 'phone_source'}`` from a TextGrid path or a ``{'words', 'phones'}`` mapping.

    Words from a path come from ``read_word_intervals`` (BAD_WORDS removed, as the semantic
    targets). ``phones``: 'textgrid' (default) takes the one tier of ``textgrid`` whose name
    contains 'phone' (or the mapping's 'phones'); a path takes that file's phone tier; a
    sequence of (start, end, label) is used as given; None (explicit) means no phone timing
    ('phones' is None and the phoneme-rate column is dropped). Words and phones get the same
    filters whatever their source. A phone source is rejected (ValueError) when no ARPAbet
    phone survives or more than PHONE_DROP_MAX of its speech labels (not NON_SPEECH) are not
    ARPAbet, e.g. IPA from an aligner; ``n_phone_labels_dropped`` counts those labels.
    """
    if isinstance(textgrid, (str, Path)):
        words = read_word_intervals(textgrid, reference_dir)
    else:
        words = textgrid['words']
    if phones is None:
        source, rows = 'none', None
    elif isinstance(phones, str) and phones == 'textgrid':
        source = 'textgrid'
        if isinstance(textgrid, (str, Path)):
            rows = _phone_tier(textgrid, reference_dir)
        elif textgrid.get('phones') is None:
            raise ValueError('Phone timing is required for the phoneme-rate column (or pass phones=None)')
        else:
            rows = textgrid['phones']
    elif isinstance(phones, (str, Path)):
        source, rows = 'file', _phone_tier(phones, reference_dir)
    else:
        source, rows = 'intervals', phones
    out = {'words': [(float(a), float(b), str(w)) for a, b, w in words if _is_word(str(w))],
           'phones': None if rows is None else [(float(a), float(b), str(p)) for a, b, p in rows
                                                if _is_phone(str(p))],
           'phone_source': source, 'n_phone_labels_dropped': None}
    if rows is not None:
        dropped = sum(not _is_phone(str(p)) and _bare(str(p)).lower() not in NON_SPEECH for *_, p in rows)
        out['n_phone_labels_dropped'] = dropped
        if not out['phones']:
            raise ValueError(f'No ARPAbet phones in the {source} phone timing (IPA labels? or pass phones=None)')
        if dropped > PHONE_DROP_MAX * (len(out['phones']) + dropped):
            raise ValueError(f'{dropped} of {len(out["phones"]) + dropped} speech phone labels in the {source} '
                             f'phone timing are not ARPAbet (IPA labels?)')
    for key in ('words', 'phones'):
        if any(not np.isfinite([a, b]).all() or b <= a for a, b, _ in out[key] or ()):
            raise ValueError(f'Invalid {key} intervals')
    if not out['words']:
        raise ValueError('No word intervals after filtering')
    return out


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX):
    """Unnormalized HTK-mel triangles [n_mels, n_fft // 2 + 1] (constant gains vanish in the story z-score)."""
    mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
    edges = 700.0 * (10.0 ** (np.linspace(mel(fmin), mel(fmax), n_mels + 2) / 2595.0) - 1.0)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    fb = np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    if (fb.sum(1) <= 0).any():
        raise ValueError('Empty mel band: increase n_fft or reduce n_mels')
    return fb


def _frames(x, length, hop=HOP):
    """Strided frame view [T, length] and frame-centre times in seconds at ``SR``."""
    if len(x) < length:
        return np.zeros((0, length)), np.zeros(0)
    view = np.lib.stride_tricks.sliding_window_view(x, length)[::hop]
    return view, (np.arange(len(view)) * hop + length / 2) / SR


def log_mel(x):
    """Per-frame log mel power [T, N_MELS], frame-centre times and the log floor, for 16-kHz samples.

    The floor is LOG_FLOOR times the story's peak band power, so a gain change only shifts
    every value by a constant (removed by the story z-score).
    """
    view, times = _frames(x, WIN)
    fb, window = mel_filterbank().T, np.hanning(WIN)
    power = np.empty((len(view), N_MELS))
    for start in range(0, len(view), CHUNK):
        power[start:start + CHUNK] = np.abs(np.fft.rfft(view[start:start + CHUNK] * window, N_FFT, axis=1)) ** 2 @ fb
    floor = LOG_FLOOR * (power.max() if power.size and power.max() > 0 else 1.0)
    return np.log(power + floor), times, float(np.log(floor))


def f0_autocorr(x):
    """Deterministic per-frame F0 (Hz, 0 when unvoiced) and frame-centre times for 16-kHz samples.

    Boersma (1993): each frame minus its window-weighted mean (a constant offset would otherwise
    look periodic at every lag), the Hann-windowed autocorrelation normalized by r(0) and divided
    by the window's own normalized autocorrelation. r is evaluated every 1/F0_UPSAMPLE sample by
    zero-padding the power spectrum (exact band-limited interpolation, as Praat's sinc-interpolated
    peak strengths), so a period between two samples keeps its full strength. Candidates are the
    local maxima of r (r[k] > r[k - 1], r[k] >= r[k + 1]) at lags within half a sample of
    [SR / F0_MAX, SR / F0_MIN], each refined parabolically (vertex lag and height); the one
    maximizing height - OCTAVE_COST * log2(F0_MIN * lag) wins and is clipped to [F0_MIN, F0_MAX].
    Voiced iff a candidate exists, its height >= VOICING and the frame's peak is at least SILENCE
    times the story peak. A period longer than 1/F0_MIN (hum, creak) has no candidate, because r
    still rises at the range edge, and the frame is unvoiced.
    """
    length = int(round(3 * SR / F0_MIN))
    view, times = _frames(x, length)
    window = np.hanning(length)
    n_fft = 1 << int(np.ceil(np.log2(2 * length)))
    up, half = F0_UPSAMPLE, F0_UPSAMPLE // 2  # Lags below are in 1/up samples.
    lags = np.arange(up * int(np.ceil(SR / F0_MAX)) - half, up * int(np.floor(SR / F0_MIN)) + half + 1)
    top = lags[-1] + 2
    rw = np.fft.irfft(np.abs(np.fft.rfft(window, n_fft)) ** 2, up * n_fft)[:top]
    rw = rw / rw[0]
    floor = SILENCE * np.abs(x).max() if len(x) else 0.0
    f0 = np.zeros(len(view))
    step = max(1, CHUNK // up)
    for start in range(0, len(view), step):
        block = view[start:start + step]
        block = block - (block @ window / window.sum())[:, None]  # Window-weighted mean: no DC 'periodicity'.
        ra = np.fft.irfft(np.abs(np.fft.rfft(block * window, n_fft, axis=1)) ** 2, up * n_fft, axis=1)[:, :top]
        energy = ra[:, :1]
        r = np.divide(ra, energy, out=np.zeros_like(ra), where=energy > 0) / rw
        a, b, c = r[:, lags - 1], r[:, lags], r[:, lags + 1]
        peak = (b > a) & (b >= c)  # Local maxima only.
        curve = a - 2 * b + c
        shift = np.clip(np.divide(0.5 * (a - c), curve, out=np.zeros_like(a), where=curve < 0), -0.5, 0.5)
        height = b - 0.25 * (a - c) * shift  # Parabola vertex height: candidates compete on peak strength.
        score = np.where(peak, height - OCTAVE_COST * np.log2(F0_MIN * (lags + shift) / (up * SR)), -np.inf)
        best, rows = np.argmax(score, axis=1), np.arange(len(block))
        lag, strength = lags[best] + shift[rows, best], height[rows, best]
        voiced = peak[rows, best] & (strength >= VOICING) & (energy[:, 0] > 0) & (np.abs(block).max(1) >= floor)
        f0[start:start + step] = np.where(voiced, np.clip(up * SR / lag, F0_MIN, F0_MAX), 0.0)
    return f0, times


def f0_track(x, method=F0_METHOD):
    """(F0 per frame with 0 when unvoiced, frame times, method) for 16-kHz samples ('autocorr' or 'pyin')."""
    if method == 'autocorr':
        f0, times = f0_autocorr(x)
    elif method == 'pyin':
        import librosa
        f0, voiced, _ = librosa.pyin(x, fmin=F0_MIN, fmax=F0_MAX, sr=SR, frame_length=PYIN_FRAME, hop_length=HOP)
        f0 = np.where(voiced & np.isfinite(f0), f0, 0.0)
        times = np.arange(len(f0)) * HOP / SR
    else:
        raise ValueError(f'Unknown F0 method {method} (registered: {F0_METHOD!r}; or pyin by name)')
    return np.asarray(f0, dtype=np.float64), np.asarray(times, dtype=np.float64), method


def _versions(method):
    """Versions of the numeric libraries the features depend on (librosa only for pyin)."""
    import scipy
    out = dict(numpy=np.__version__, scipy=scipy.__version__)
    if method == 'pyin':
        import librosa
        out['librosa'] = librosa.__version__
    return out


def _spans(times, left, right):
    """[lo, hi) index ranges of sorted ``times`` inside each [left, right)."""
    return np.searchsorted(times, left, 'left'), np.searchsorted(times, right, 'left')


def _count(points, left, right):
    points = np.asarray(points, dtype=np.float64)
    return ((points[None] >= left[:, None]) & (points[None] < right[:, None])).sum(1).astype(np.float64)


def acoustic_features(story_audio, textgrid, centers, tr_seconds, *, phones='textgrid', f0_method=F0_METHOD,
                      reference_dir=None):
    """Raw (unstandardized) float64 features [len(centers), F] and a JSON-ready info dict.

    ``centers`` are stimulus-relative TR centres in seconds (``tr_centers``); TR t covers
    [centers[t] - tr_seconds / 2, centers[t] + tr_seconds / 2). ``phones`` as in
    ``read_timing``; with None the phoneme-rate column is dropped (``info['feature_names']``
    lists the columns). Raises ValueError when the words or phones run more than END_TOLERANCE
    seconds past the end of the audio or a timing column has no word or phone in any TR.
    """
    tr_seconds = float(tr_seconds)
    if not np.isfinite(tr_seconds) or tr_seconds <= 0:
        raise ValueError('TR duration must be finite and positive')
    centers = np.asarray(centers, dtype=np.float64)
    left, right = centers - tr_seconds / 2, centers + tr_seconds / 2
    timing = read_timing(textgrid, reference_dir, phones)
    raw, rate = load_audio(story_audio)
    x = resample(raw, rate)
    duration = len(raw) / rate
    last = max(b for _, b, _ in timing['words'])
    last_phone = None if timing['phones'] is None else max(b for _, b, _ in timing['phones'])
    for kind, end in (('Transcript', last), ('Phone timing', last_phone)):
        if end is not None and end > duration + END_TOLERANCE:
            raise ValueError(f'{kind} ends at {end:.2f} s but the audio lasts {duration:.2f} s')
    out = np.zeros((len(centers), len(FEATURE_NAMES)))
    mel, mel_times, mel_floor = log_mel(x)
    f0, f0_times, method = f0_track(x, f0_method)
    mel_lo, mel_hi = _spans(mel_times, left, right)
    f0_lo, f0_hi = _spans(f0_times, left, right)
    sample_lo, sample_hi = (np.clip(np.ceil(edge * SR), 0, len(x)).astype(np.int64) for edge in (left, right))
    for t in range(len(centers)):
        frames, segment = mel[mel_lo[t]:mel_hi[t]], x[sample_lo[t]:sample_hi[t]]
        out[t, :N_MELS] = frames.mean(0) if len(frames) else mel_floor
        out[t, N_MELS] = np.sqrt(np.mean(segment ** 2)) if len(segment) else 0.0
        voiced = f0[f0_lo[t]:f0_hi[t]]
        voiced = voiced[voiced > 0]
        out[t, N_MELS + 1] = np.median(voiced) if len(voiced) else 0.0
    words, phone_rows = timing['words'], timing['phones']
    out[:, N_MELS + 2] = _count([(a + b) / 2 for a, b, _ in words], left, right) / tr_seconds
    if phone_rows is not None:
        out[:, N_MELS + 3] = _count([(a + b) / 2 for a, b, _ in phone_rows], left, right) / tr_seconds
    out[:, N_MELS + 4] = _count([a for a, _, _ in words], left, right)
    for col, kind in ((N_MELS + 2, 'word'), (N_MELS + 3, 'phone')):
        if (kind == 'word' or phone_rows is not None) and not out[:, col].any():
            raise ValueError(f'No {kind} midpoint falls in any TR (timing from another story?)')
    names = list(FEATURE_NAMES)
    if phone_rows is None:
        out, names = np.delete(out, N_MELS + 3, axis=1), [n for n in names if n != 'phoneme_rate']
    if not np.isfinite(out).all():
        raise ValueError('Non-finite acoustic features')
    info = dict(feature_names=names, f0_method=method, versions=_versions(method), phone_source=timing['phone_source'],
                input_sample_rate=rate, sample_rate=SR, audio_seconds=duration, last_word_end=last,
                last_phone_end=last_phone, mel_log_floor=mel_floor, n_words=len(words),
                n_phones=None if phone_rows is None else len(phone_rows),
                n_phone_labels_dropped=timing['n_phone_labels_dropped'],
                voiced_frame_fraction=float((f0 > 0).mean()) if len(f0) else 0.0, n_tr=len(centers),
                tr_seconds=tr_seconds, first_center=float(centers[0]) if len(centers) else None)
    return out, info


def zscore(features):
    """Column z-score over the story's TRs; columns with std <= STD_FLOOR become zero."""
    features = np.asarray(features, dtype=np.float64)
    centered = features - features.mean(0)
    std = features.std(0)
    keep = std > STD_FLOOR
    return np.where(keep, centered / np.where(keep, std, 1.0), 0.0)


def acoustic_targets(story_audio, textgrid, n_tr, tr_start, tr_seconds, *, phones='textgrid', f0_method=F0_METHOD,
                     reference_dir=None, return_info=False):
    """Per-TR P-AC features [n_tr, F], float64, z-scored per story (optionally with the info dict).

    ``story_audio`` is a WAV path (``ds003020/stimuli/<story>.wav``) or ``(samples, rate)``;
    ``textgrid`` a TextGrid path or a ``{'words', 'phones'}`` mapping; ``phones`` as in
    ``read_timing``. ``tr_start`` and ``tr_seconds`` are the manifest's ``tr_start_seconds``
    (11.0) and ``tr_seconds`` (2.0).
    """
    run = dict(run_id='acoustic', n_tr=int(n_tr), tr_start_seconds=tr_start, tr_seconds=tr_seconds)
    raw, info = acoustic_features(story_audio, textgrid, tr_centers(run, None), tr_seconds, phones=phones,
                                  f0_method=f0_method, reference_dir=reference_dir)
    return (zscore(raw), info) if return_info else zscore(raw)


def _manifest_dir(manifest_path):
    """Directory relative run paths resolve against, as in ``run_words`` and the target builder."""
    manifest_dir = Path(manifest_path).resolve().parent
    source = (json.loads(Path(manifest_path).read_text()).get('neural_manifest_path')
              if Path(manifest_path).is_file() else None)
    return Path(source).resolve().parent if source else manifest_dir


def _resolve(path, manifest_dir):
    path = Path(path)
    return path if path.is_absolute() else Path(manifest_dir) / path


def run_acoustic_targets(run, manifest_path, story_audio, *, phones='textgrid', f0_method=F0_METHOD,
                         reference_dir=None, return_info=False):
    """``acoustic_targets`` for a manifest run: its TextGrid, TR centres and a relative ``phones``
    path resolved as in ``run_words``."""
    manifest_dir = _manifest_dir(manifest_path)
    if isinstance(phones, (str, Path)) and not (isinstance(phones, str) and phones == 'textgrid'):
        phones = _resolve(phones, manifest_dir)
    raw, info = acoustic_features(story_audio, _resolve(run['stimulus_path'], manifest_dir),
                                  tr_centers(run, manifest_dir), run['tr_seconds'], phones=phones,
                                  f0_method=f0_method, reference_dir=reference_dir)
    info.update(run_id=run['run_id'], story=run.get('story'),
                phone_path=str(phones) if isinstance(phones, Path) else None)
    return (zscore(raw), info) if return_info else zscore(raw)


def story_wav(story, run, manifest_path, audio_root=None):
    """The story's WAV: ``<audio_root>/<story>.wav`` when given, else ``run['audio_path']`` (LPP
    manifests), else ds003020's ``<dataset>/stimuli/<story>.wav`` for a TextGrid at
    ``<dataset>/derivatives/TextGrids/<story>.TextGrid``."""
    manifest_dir = _manifest_dir(manifest_path)
    if audio_root is not None:
        path = Path(audio_root) / f'{story}.wav'
    elif run.get('audio_path'):
        path = _resolve(run['audio_path'], manifest_dir)
    else:
        grid = _resolve(run['stimulus_path'], manifest_dir)
        if grid.parent.name != 'TextGrids':
            raise FileNotFoundError(f'{story}: no audio_path in the run and no audio_root for {grid}')
        path = grid.parents[2] / 'stimuli' / f'{story}.wav'
    if not path.is_file():
        raise FileNotFoundError(f'{story}: audio not found at {path} (fetch ds003020 stimuli/<story>.wav)')
    return path


def story_features(story, run, manifest_path, *, audio_root=None, phones='textgrid', f0_method=F0_METHOD,
                   reference_dir=None, return_info=False):
    """Runner entry point: z-scored P-AC features [n_tr, F] of one budget story (optionally with info).

    Resolves the WAV with ``story_wav`` and the TextGrid and TR centres from the manifest run;
    ``info`` adds the WAV path and SHA-256 for the cell record. The caller logs the audio and
    transcript reads (rule R1) and passes budget stories only.
    """
    if run.get('story', story) != story:
        raise ValueError(f'Run {run.get("run_id")} is story {run.get("story")}, not {story}')
    path = story_wav(story, run, manifest_path, audio_root)
    out = run_acoustic_targets(run, manifest_path, path, phones=phones, f0_method=f0_method,
                               reference_dir=reference_dir, return_info=return_info)
    if not return_info:
        return out
    z, info = out
    info.update(audio_path=str(path), audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return z, info


def acoustic_rows(features_by_run, run_ids, target_trs, rows=None, fill=0.0):
    """Row-aligned predictors [N, F]: each row's story features at its target TR (the W1 target clock).

    ``features_by_run`` is keyed like ``run_ids`` (run ids or stories). With ``rows`` (e.g. the
    fold's ``fit_pos``), only those positions are looked up and every other row is ``fill``:
    ``encoding_scores`` reads predictors at its ``rows`` only, so no audio outside the fitting
    part is needed.
    """
    if len(run_ids) != len(target_trs):
        raise ValueError('run_ids and target_trs must have the same length')
    use = range(len(run_ids)) if rows is None else sorted({int(i) for i in rows})
    if not len(use) or use[0] < 0 or use[-1] >= len(run_ids):
        raise ValueError('rows must be non-empty positions into run_ids')
    out = None
    for i in use:
        run_id, t = run_ids[i], target_trs[i]
        features = np.asarray(features_by_run[run_id], dtype=np.float64)
        if int(t) != t or not 0 <= t < len(features):
            raise ValueError(f'Target TR {t} lies outside run {run_id}')
        if out is None:
            out = np.full((len(run_ids), features.shape[1]), float(fill))
        elif features.shape[1] != out.shape[1]:
            raise ValueError(f'Run {run_id} has {features.shape[1]} features, expected {out.shape[1]}')
        out[i] = features[int(t)]
    return out


def acoustic_scores(features, design, rows, row_groups, folds, alphas, device):
    """P-AC nested out-of-fold r per design column: ``encoding_scores`` with acoustic predictors.

    ``features`` [N, F] is row-aligned with ``design`` [N, C] (``acoustic_rows``; rows outside
    ``rows`` may be fill); the other arguments are W1's. Returns (r per column, alpha, mean-r curve).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != design.shape[0] or not np.isfinite(features).all():
        raise ValueError('Acoustic predictors must be finite and row-aligned with the design')
    return encoding_scores(features, design, rows, row_groups, folds, alphas, device)
