"""PROTOCOL_V4 P-AC: a synthetic tone and click track with synthetic TextGrids and a fake dataset tree."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from src.encoding_weights import encoding_scores
from src.priors import acoustic as ac
from src.transcripts import run_words

SR = ac.SR
TR, TR_START, N_TR = 2.0, 3.0, 7  # Bins [2, 4), [4, 6), ..., [14, 16) seconds.
DURATION = 16.0
F0_COL, RMS_COL, WORD_COL, PHONE_COL, ONSET_COL = (ac.FEATURE_NAMES.index(n) for n in (
    'f0', 'rms', 'word_rate', 'phoneme_rate', 'word_onset_count'))
WORDS = [(1.0, 1.9, 'pre'), (2.0, 2.6, 'alpha'), (3.6, 4.6, 'beta'), (4.6, 5.0, 'sp'), (5.0, 5.5, 'gamma'),
         (5.5, 6.5, 'delta'), (9.0, 9.4, 'eps')]  # 'delta' has its midpoint exactly on the 6-s bin edge.
PHONES = [(1.0, 1.9, 'P'), (2.0, 2.2, 'AE1'), (2.2, 2.4, 'L'), (2.4, 2.6, 'F'), (3.6, 4.0, 'B'), (4.0, 4.6, 'EY1'),
          (4.6, 5.0, 'sp'), (5.0, 5.2, 'G'), (5.2, 5.5, 'AE1'), (5.5, 5.9, 'D'), (5.9, 6.2, 'EH1'), (6.2, 6.35, 'L'),
          (6.35, 6.5, 'T'), (7.0, 7.5, 'lg'), (9.0, 9.2, 'EH1'), (9.2, 9.4, 'P')]
# Hand counts per bin: word midpoints, phone midpoints (no 'sp'/'lg'/''), word onsets.
WORD_COUNTS = [1, 2, 1, 1, 0, 0, 0]
PHONE_COUNTS = [4, 4, 3, 2, 0, 0, 0]
ONSET_COUNTS = [2, 2, 0, 1, 0, 0, 0]
SILENT, TONE200, TONE120, CLICKS = (0, 4, 6), (1, 5), 3, 2


def track(rate=SR):
    """A loud tone before the first bin, 200/120/200 Hz tones and a click train, 0.2-s margins in each bin."""
    t = np.arange(int(DURATION * rate)) / rate
    tone = lambda f, amp, a, b: np.where((t >= a) & (t < b), amp * np.sin(2 * np.pi * f * t), 0.0)
    x = tone(300, 1.0, 0.0, 1.8) + tone(200, 0.5, 4.2, 5.8) + tone(120, 0.25, 8.2, 9.8) + tone(200, 0.5, 12.2, 13.8)
    x[np.round(np.arange(6.2, 7.8, 0.25) * rate).astype(int)] = 0.9
    return x


def timing():
    return {'words': WORDS, 'phones': PHONES}


def tier(name, rows, index):
    """An IntervalTier covering [0, DURATION] with gaps filled by empty intervals."""
    filled, last = [], 0.0
    for a, b, w in rows:
        if a > last:
            filled.append((last, a, ''))
        filled.append((a, b, w))
        last = b
    filled.append((last, DURATION, ''))
    return (f'    item [{index}]:\n        class = "IntervalTier" \n        name = "{name}" \n'
            f'        xmin = 0 \n        xmax = {DURATION} \n        intervals: size = {len(filled)} \n' +
            ''.join(f'        intervals [{i}]:\n            xmin = {a} \n            xmax = {b} \n'
                    f'            text = "{w}" \n' for i, (a, b, w) in enumerate(filled, 1)))


def textgrid(path, tiers=(('phone', PHONES), ('word', WORDS)), encoding='utf-8'):
    """A Praat TextGrid with the given tiers (UTF-16 with a BOM as the official LPP files)."""
    path.write_text(f'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0 \nxmax = {DURATION} \n'
                    f'tiers? <exists> \nsize = {len(tiers)} \nitem []: \n' +
                    ''.join(tier(name, rows, i) for i, (name, rows) in enumerate(tiers, 1)), encoding=encoding)
    return path


@pytest.fixture
def reference_parser():
    root = Path(__file__).resolve().parents[1] / 'reference/semantic-decoding'
    if not (root / 'decoding/utils_ridge/textgrid.py').is_file():
        pytest.skip('Parser-dependent test: local Tang/Huth TextGrid parser is unavailable')
    return root


def centers():
    return TR_START + np.arange(N_TR) * TR


def raw(audio=None, method='autocorr'):
    """Unstandardized features of ``(samples, rate)`` (default: the 16-kHz track) with in-memory timing."""
    return ac.acoustic_features((track(), SR) if audio is None else audio, timing(), centers(), TR, f0_method=method)


def test_tone_and_click_track_features():
    x = track()
    out, info = raw((x, SR))
    assert out.shape == (N_TR, len(ac.FEATURE_NAMES)) == (N_TR, 37) and out.dtype == np.float64
    assert info['f0_method'] == 'autocorr' and info['n_words'] == 6 and info['n_phones'] == 14
    # F0: median voiced frequency, 0 when unvoiced (silence, clicks, and the tone before the first bin).
    f0 = out[:, F0_COL]
    np.testing.assert_allclose(f0[list(TONE200)], 200.0, atol=1.0)
    assert abs(f0[TONE120] - 120.0) < 1.5
    assert (f0[list(SILENT) + [CLICKS]] == 0).all()
    # RMS over exactly the samples of each half-open bin.
    for t, c in enumerate(centers()):
        segment = x[int(round((c - TR / 2) * SR)):int(round((c + TR / 2) * SR))]
        np.testing.assert_allclose(out[t, RMS_COL], np.sqrt(np.mean(segment ** 2)), rtol=1e-10, atol=1e-15)
    # Mel: silence sits at the log floor; tones peak in their own band; clicks are broadband.
    mel = out[:, :ac.N_MELS]
    np.testing.assert_allclose(mel[list(SILENT)], info['mel_log_floor'], rtol=1e-13)
    fb = ac.mel_filterbank()
    band_centers = np.arange(fb.shape[1])[fb.argmax(1)] * SR / ac.N_FFT
    assert abs(band_centers[mel[1].argmax()] - 200) < 60 and abs(band_centers[mel[TONE120].argmax()] - 120) < 60
    assert mel[1, :8].mean() > mel[1, -8:].mean() + 5 and mel[CLICKS, -8:].mean() > mel[CLICKS, :8].mean()
    assert mel[CLICKS, -1] > mel[1, -1]
    np.testing.assert_allclose(out[1, :RMS_COL + 2], out[5, :RMS_COL + 2], rtol=1e-6, atol=1e-6)  # Time-invariant.
    # Timing: the semantic-target bins (midpoints, half-open), filtered words and ARPAbet phones.
    np.testing.assert_array_equal(out[:, WORD_COL], np.array(WORD_COUNTS) / TR)
    np.testing.assert_array_equal(out[:, PHONE_COL], np.array(PHONE_COUNTS) / TR)
    np.testing.assert_array_equal(out[:, ONSET_COL], ONSET_COUNTS)


def test_textgrid_and_wav_paths_match_in_memory_inputs(tmp_path, reference_parser):
    grid = textgrid(tmp_path / 'story.TextGrid')
    wav = tmp_path / 'story.wav'
    wavfile.write(wav, SR, np.round(track() * 32767).astype(np.int16))
    loaded = ac.read_timing(grid, reference_parser)
    assert [w for *_, w in loaded['words']] == ['pre', 'alpha', 'beta', 'gamma', 'delta', 'eps']
    assert len(loaded['phones']) == 14
    from_paths, _ = ac.acoustic_features(wav, grid, centers(), TR, f0_method='autocorr', reference_dir=reference_parser)
    in_memory, _ = ac.acoustic_features(ac.load_audio(wav), loaded, centers(), TR, f0_method='autocorr')
    np.testing.assert_array_equal(from_paths, in_memory)
    np.testing.assert_allclose(from_paths[:, RMS_COL:], raw()[0][:, RMS_COL:], atol=1e-2)  # int16 rounding only.
    # Word-rate bins are exactly the bins of the semantic targets (transcripts.run_words / downsample_words).
    run = dict(run_id='r', stimulus_path=str(grid), n_tr=N_TR, tr_start_seconds=TR_START, tr_seconds=TR)
    rows = run_words(run, tmp_path / 'manifest.json', reference_parser)
    counts = np.bincount([r['bin_index'] for r in rows if r['bin_index'] is not None], minlength=N_TR)
    np.testing.assert_array_equal(from_paths[:, WORD_COL] * TR, counts)


def test_resampled_input_keeps_pitch_and_alignment():
    out, info = raw((track(22050), 22050))
    assert info['input_sample_rate'] == 22050 and info['sample_rate'] == SR
    np.testing.assert_allclose(out[list(TONE200), F0_COL], 200.0, atol=1.0)
    assert abs(out[TONE120, F0_COL] - 120.0) < 1.5 and (out[list(SILENT), F0_COL] == 0).all()
    tonal = [t for t in range(N_TR) if t != CLICKS]  # Resampling low-passes the broadband clicks.
    np.testing.assert_allclose(out[tonal, RMS_COL], raw()[0][tonal, RMS_COL], rtol=1e-3, atol=1e-6)


def test_acoustic_targets_are_story_zscored_and_deterministic():
    z, info = ac.acoustic_targets((track(), SR), timing(), N_TR, TR_START, TR, f0_method='autocorr', return_info=True)
    again = ac.acoustic_targets((track(), SR), timing(), N_TR, TR_START, TR, f0_method='autocorr')
    np.testing.assert_array_equal(z, again)
    np.testing.assert_array_equal(z, ac.zscore(raw()[0]))
    assert z.shape == (N_TR, 37) and z.dtype == np.float64 and info['first_center'] == TR_START
    assert json.loads(json.dumps(info)) == info  # JSON-ready for the cell record.
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-12)
    np.testing.assert_allclose(ac.acoustic_targets((0.3 * track(), SR), timing(), N_TR, TR_START, TR,
                                                   f0_method='autocorr'), z, atol=1e-9)  # Gain invariance.
    flat = ac.zscore(np.column_stack([np.full(5, -23.0), np.arange(5.0)]))
    assert (flat[:, 0] == 0).all() and abs(flat[:, 1].std() - 1) < 1e-12


def test_run_acoustic_targets_resolves_manifest_paths(tmp_path, reference_parser):
    (tmp_path / 'stimuli').mkdir()
    textgrid(tmp_path / 'stimuli' / 'story.TextGrid')
    manifest = tmp_path / 'semantic' / 'semantic_manifest.json'
    manifest.parent.mkdir()
    neural = tmp_path / 'neural_manifest.json'
    neural.write_text('{}')
    manifest.write_text(json.dumps(dict(neural_manifest_path=str(neural))))
    run = dict(run_id='story_run', story='story', stimulus_path='stimuli/story.TextGrid', n_tr=N_TR,
               tr_start_seconds=TR_START, tr_seconds=TR)
    z, info = ac.run_acoustic_targets(run, manifest, (track(), SR), f0_method='autocorr',
                                      reference_dir=reference_parser, return_info=True)
    expected = ac.acoustic_targets((track(), SR), timing(), N_TR, TR_START, TR, f0_method='autocorr')
    np.testing.assert_array_equal(z, expected)
    assert info['run_id'] == 'story_run' and info['story'] == 'story'


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match='Transcript ends'):
        ac.acoustic_features((track()[:SR * 8], SR), timing(), centers(), TR)  # Audio stops before 'eps'.
    with pytest.raises(ValueError, match='Phone timing'):
        ac.acoustic_features((track(), SR), {'words': WORDS}, centers(), TR)
    with pytest.raises(ValueError, match='Unknown F0'):
        raw(method='yin')
    with pytest.raises(ValueError, match='TR duration'):
        ac.acoustic_features((track(), SR), timing(), centers(), 0.0)


def test_mel_filterbank_bands():
    fb = ac.mel_filterbank()
    assert fb.shape == (32, ac.N_FFT // 2 + 1) and (fb.sum(1) > 0).all() and (fb >= 0).all()
    peaks = fb.argmax(1)
    assert (np.diff(peaks) >= 0).all() and peaks[-1] * SR / ac.N_FFT <= ac.MEL_FMAX


def test_acoustic_rows_and_scores_are_w1_encoding_scores():
    rng = np.random.default_rng(0)
    by_run = {'a': rng.normal(size=(40, 5)), 'b': rng.normal(size=(36, 5))}
    run_ids, trs = ['a'] * 30 + ['b'] * 30, list(range(5, 35)) + list(range(2, 32))
    x = ac.acoustic_rows(by_run, run_ids, trs)
    assert np.array_equal(x[0], by_run['a'][5]) and np.array_equal(x[30], by_run['b'][2])
    design = np.column_stack([2 * x[:, 1] + 0.3 * rng.normal(size=60), rng.normal(size=60)])
    design = torch.as_tensor((design - design.mean(0)) / design.std(0))
    groups = [f'{r}:{int(i % 30 >= 15)}' for i, r in enumerate(run_ids)]  # Half-story groups.
    folds = [('a:0',), ('a:1',), ('b:0',), ('b:1',)]
    alphas = np.logspace(-1, 3, 5)
    r, alpha, curve = ac.acoustic_scores(x, design, list(range(60)), groups, folds, alphas, 'cpu')
    r0, alpha0, curve0 = encoding_scores(x, design, list(range(60)), groups, folds, alphas, 'cpu')
    np.testing.assert_array_equal(r, r0)
    np.testing.assert_array_equal(curve, curve0)
    assert alpha == alpha0 and r[0] > 0.8 and abs(r[1]) < 0.4
    with pytest.raises(ValueError, match='outside run'):
        ac.acoustic_rows(by_run, ['b'], [36])
    with pytest.raises(ValueError, match='row-aligned'):
        ac.acoustic_scores(x[:-1], design, list(range(59)), groups, folds, alphas, 'cpu')


def test_f0_below_range_is_unvoiced_and_in_range_f0_is_clipped():
    t = np.arange(2 * SR) / SR
    pulses = lambda hz: (np.arange(2 * SR) % round(SR / hz) == 0).astype(float)
    for x in (np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 60 * t), pulses(40), pulses(50)):  # Hum and creak.
        f0, _ = ac.f0_autocorr(x)
        assert len(f0) and (f0 == 0).all()
    for hz, expected in ((65, 65), (100, 100), (499, 499), (504, ac.F0_MAX)):
        f0, _ = ac.f0_autocorr(np.sin(2 * np.pi * hz * t))
        assert (f0 > 0).all() and np.abs(f0 - expected).max() < 0.5
    f0, _ = ac.f0_autocorr(pulses(100))
    assert (f0 > 0).all() and np.abs(f0 - 100).max() < 0.5
    # A 50-Hz hum at 5% of the story peak leaves pauses unvoiced (it read ~508 Hz before) and tones intact.
    hum = track() + 0.05 * np.sin(2 * np.pi * 50 * np.arange(int(DURATION * SR)) / SR)
    out, _ = raw((hum, SR))
    assert (out[list(SILENT) + [CLICKS], F0_COL] == 0).all()
    np.testing.assert_allclose(out[list(TONE200), F0_COL], 200.0, atol=1.0)
    assert abs(out[TONE120, F0_COL] - 120.0) < 1.5


def test_f0_method_is_registered_not_environment_dependent(monkeypatch):
    real = importlib.util.find_spec  # Pretend librosa is installed: the default must not change.
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a: real('json' if name == 'librosa' else name, *a))
    out, info = ac.acoustic_features((track(), SR), timing(), centers(), TR)  # Defaults, "librosa installed".
    assert ac.F0_METHOD == 'autocorr' and info['f0_method'] == 'autocorr'
    assert set(info['versions']) == {'numpy', 'scipy'} and info['versions']['numpy'] == np.__version__
    np.testing.assert_array_equal(out, raw()[0])
    with pytest.raises(ValueError, match='Unknown F0'):
        raw(method='auto')


def test_phone_sources_and_explicit_reduced_set(tmp_path, reference_parser):
    full, info = raw()
    assert info['phone_source'] == 'textgrid' and info['n_phones'] == 14
    words_only = {'words': WORDS}
    reduced, rinfo = ac.acoustic_features((track(), SR), words_only, centers(), TR, phones=None)
    assert reduced.shape == (N_TR, 36) and rinfo['phone_source'] == 'none' and rinfo['n_phones'] is None
    assert rinfo['feature_names'] == [n for n in ac.FEATURE_NAMES if n != 'phoneme_rate']
    np.testing.assert_array_equal(reduced, np.delete(full, PHONE_COL, axis=1))
    listed, linfo = ac.acoustic_features((track(), SR), words_only, centers(), TR, phones=PHONES)
    np.testing.assert_array_equal(listed, full)
    assert linfo['phone_source'] == 'intervals'
    z = ac.acoustic_targets((track(), SR), words_only, N_TR, TR_START, TR, phones=None)
    np.testing.assert_array_equal(z, np.delete(ac.zscore(full), PHONE_COL, axis=1))
    # LPP-like files: a word-only TextGrid and a separate UTF-16 phone TextGrid.
    grid = textgrid(tmp_path / 'words.TextGrid', (('words', WORDS),))
    phone_grid = textgrid(tmp_path / 'phones.TextGrid', (('phones', PHONES),), encoding='utf-16')
    with pytest.raises(ValueError, match='phone tier'):
        ac.acoustic_features((track(), SR), grid, centers(), TR, reference_dir=reference_parser)
    from_file, finfo = ac.acoustic_features((track(), SR), grid, centers(), TR, phones=phone_grid,
                                            reference_dir=reference_parser)
    np.testing.assert_array_equal(from_file, full)
    assert finfo['phone_source'] == 'file'
    none, _ = ac.acoustic_features((track(), SR), grid, centers(), TR, phones=None, reference_dir=reference_parser)
    np.testing.assert_array_equal(none, reduced)


def dataset_tree(root):
    """A fake ds003020 tree: derivatives/TextGrids/story.TextGrid, stimuli/story.wav, a manifest."""
    grids = root / 'ds003020' / 'derivatives' / 'TextGrids'
    grids.mkdir(parents=True)
    (root / 'ds003020' / 'stimuli').mkdir()
    textgrid(grids / 'story.TextGrid')
    wav = root / 'ds003020' / 'stimuli' / 'story.wav'
    wavfile.write(wav, SR, np.round(track() * 32767).astype(np.int16))
    manifest = root / 'semantic_manifest.json'
    manifest.write_text('{}')
    run = dict(run_id='story_run', story='story', stimulus_path=str(grids / 'story.TextGrid'), n_tr=N_TR,
               tr_start_seconds=TR_START, tr_seconds=TR)
    return run, manifest, wav


def test_story_features_resolves_the_story_wav(tmp_path, reference_parser):
    run, manifest, wav = dataset_tree(tmp_path)
    z = ac.story_features('story', run, manifest, reference_dir=reference_parser)  # The runner's call.
    expected = ac.run_acoustic_targets(run, manifest, wav, reference_dir=reference_parser)
    np.testing.assert_array_equal(z, expected)
    assert ac.story_wav('story', run, manifest) == wav
    z2, info = ac.story_features('story', run, manifest, reference_dir=reference_parser, return_info=True)
    np.testing.assert_array_equal(z2, z)
    assert info['audio_path'] == str(wav) and info['f0_method'] == 'autocorr' and info['story'] == 'story'
    assert len(info['audio_sha256']) == 64 and json.loads(json.dumps(info)) == info
    # LPP manifests carry audio_path (relative paths resolve against the manifest); audio_root overrides.
    (tmp_path / 'lpp').mkdir()
    lpp_wav = tmp_path / 'lpp' / 'section-1.wav'
    lpp_wav.write_bytes(wav.read_bytes())
    lpp = dict(run, story='1', audio_path='lpp/section-1.wav')
    assert ac.story_wav('1', lpp, manifest) == lpp_wav.resolve()
    assert ac.story_wav('story', lpp, manifest, audio_root=wav.parent) == wav
    with pytest.raises(FileNotFoundError, match='audio not found'):
        ac.story_wav('other', dict(run, story='other'), manifest)
    with pytest.raises(FileNotFoundError, match='no audio_path'):
        ac.story_wav('story', dict(run, stimulus_path=str(tmp_path / 'story.TextGrid')), manifest)
    with pytest.raises(ValueError, match='not other'):
        ac.story_features('other', run, manifest)


def test_runner_calls_match_the_module_api():
    """Every ``acm.<name>(...)`` call in scripts/run_v4.py names a function here and binds to its signature
    (a static check only; ``test_runner_pac_arm_runs_with_a_held_out_story`` runs the P-AC arm)."""
    import ast
    import inspect
    runner = Path(__file__).resolve().parents[1] / 'scripts' / 'run_v4.py'
    if not runner.is_file():
        pytest.skip('scripts/run_v4.py is absent')
    calls = [n for n in ast.walk(ast.parse(runner.read_text())) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'acm']
    if not calls:
        pytest.skip('run_v4.py does not call the acoustic module')
    for call in calls:
        assert hasattr(ac, call.func.attr), f'run_v4.py calls missing acm.{call.func.attr}'
        if not any(isinstance(a, ast.Starred) for a in call.args) and all(k.arg for k in call.keywords):
            inspect.signature(getattr(ac, call.func.attr)).bind(*call.args, **{k.arg: k.value for k in call.keywords})


def test_acoustic_rows_fill_rows_outside_the_fitting_part():
    rng = np.random.default_rng(1)
    by_story = {'a': rng.normal(size=(40, 4)), 'b': rng.normal(size=(36, 4))}  # 'c' is held out: never read.
    stories = ['a'] * 20 + ['b'] * 20 + ['c'] * 10
    trs = list(range(10, 30)) + list(range(5, 25)) + list(range(10))
    fit = np.arange(40)
    x = ac.acoustic_rows(by_story, stories, trs, rows=fit)
    assert x.shape == (50, 4) and (x[40:] == 0).all() and np.array_equal(x[25], by_story['b'][10])
    np.testing.assert_array_equal(x[:40], ac.acoustic_rows(by_story, stories[:40], trs[:40]))
    with pytest.raises(KeyError):
        ac.acoustic_rows(by_story, stories, trs)
    design = torch.as_tensor(rng.normal(size=(50, 3)))
    groups = [f'{s}:{int(i % 20 >= 10)}' for i, s in enumerate(stories)]
    folds = [('a:0',), ('a:1',), ('b:0',), ('b:1',)]
    alphas = np.logspace(-1, 3, 5)
    r, alpha, _ = ac.acoustic_scores(x, design, list(fit), groups, folds, alphas, 'cpu')
    other = x.copy()
    other[40:] = rng.normal(size=(10, 4))  # Any filler gives the same scores.
    r2, alpha2, _ = ac.acoustic_scores(other, design, list(fit), groups, folds, alphas, 'cpu')
    np.testing.assert_array_equal(r, r2)
    assert alpha == alpha2
    with pytest.raises(ValueError, match='rows must be'):
        ac.acoustic_rows(by_story, stories, trs, rows=[50])


@pytest.mark.skipif(importlib.util.find_spec('librosa') is None, reason='librosa is not installed')
def test_pyin_path_by_name():
    out, info = raw(method='pyin')
    assert info['f0_method'] == 'pyin' and 'librosa' in info['versions']
    np.testing.assert_allclose(out[list(TONE200), F0_COL], 200.0, atol=2.0)
    assert (out[list(SILENT), F0_COL] == 0).all()


def vowel(hz, seconds=1.0, formants=((700, 80), (1220, 90), (2600, 120))):
    """A steady harmonic complex (harmonics to 7 kHz) through second-order formant resonators."""
    from scipy.signal import lfilter
    phase = 2 * np.pi * hz * np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(h * phase) for h in range(1, int(7000 // hz) + 1))
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        x = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], x)
    return x / np.abs(x).max()


def test_f0_between_samples_is_not_an_octave_low():
    """Periods between two samples (230 Hz: 69.57 samples) used to lose to an integer-lag subharmonic."""
    for hz in (135, 185, 230, 260, 305, 385):  # Each read an octave low before (115 Hz at 230 Hz).
        f0, _ = ac.f0_autocorr(vowel(hz))
        assert (f0 > 0).all() and np.abs(f0 - hz).max() < 1.0, hz
    t = np.arange(SR) / SR
    for hz in (145, 220, 240, 300, 425):  # Flat spectrum to 8 kHz: the narrowest autocorrelation peaks.
        f0, _ = ac.f0_autocorr(sum(np.sin(2 * np.pi * h * hz * t) for h in range(1, int(7900 // hz) + 1)))
        assert (f0 > 0).all() and np.abs(f0 - hz).max() < 1.0, hz


def test_phone_sources_are_checked():
    words_only = {'words': WORDS}
    run = lambda phones: ac.acoustic_features((track(), SR), words_only, centers(), TR, phones=phones)
    _, info = raw()
    assert info['n_phone_labels_dropped'] == 0 and info['last_phone_end'] == 9.4
    ipa = [(a, b, 'ə') if ac._is_phone(p) else (a, b, p) for a, b, p in PHONES]  # MFA english_mfa labels.
    with pytest.raises(ValueError, match='No ARPAbet phones'):
        run(ipa)
    with pytest.raises(ValueError, match='not ARPAbet'):
        run(ipa[:-1] + PHONES[-1:])  # One ARPAbet phone among IPA labels.
    with pytest.raises(ValueError, match='Phone timing ends'):
        run([(a + 500, b + 500, p) for a, b, p in PHONES])  # Another story's phones.
    with pytest.raises(ValueError, match='No phone midpoint'):
        run([(0.2, 0.4, 'AH0'), (1.2, 1.4, 'AH0')])  # Inside the audio, before the first TR.
    stray, noise = [(15.6, 15.7, 'AHN')], [(15.0, 15.1, 'sil'), (15.1, 15.2, '{NS}'), (15.2, 15.3, 'cg')] * 9
    with pytest.raises(ValueError, match='1 of 15 speech'):
        run(PHONES + noise + stray)  # 1/15 > PHONE_DROP_MAX; silence and noise labels do not count.
    out, ninfo = run(PHONES + PHONES + noise + stray)  # 1/29: kept and counted, as ds003020's rare typos.
    assert ninfo['n_phone_labels_dropped'] == 1 and ninfo['n_phones'] == 28
    np.testing.assert_array_equal(out[:, PHONE_COL], 2 * np.array(PHONE_COUNTS) / TR)


def test_runner_pac_arm_runs_with_a_held_out_story(monkeypatch):
    """``run_v4.Priors.acoustic`` in an inner decoder fold: the budget rows include the held-out story 'c',
    whose audio must never be read; the scores are W1's ``encoding_scores`` on the fitting rows' features,
    and the per-story info (audio SHA-256, F0 method) is requested for the cell record."""
    from types import SimpleNamespace
    runner = Path(__file__).resolve().parents[1] / 'scripts' / 'run_v4.py'
    if not runner.is_file():
        pytest.skip('scripts/run_v4.py is absent')
    spec = importlib.util.spec_from_file_location('run_v4_under_test', runner)
    rv4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv4)
    rng = np.random.default_rng(2)
    feats = {'a': rng.normal(size=(30, 6)), 'b': rng.normal(size=(26, 6)), 'c': rng.normal(size=(20, 6))}
    calls, reads = [], []

    def story_features(story, run, manifest_path, return_info=False, **kw):
        calls.append((story, return_info))
        info = dict(story=story, audio_sha256='0' * 64, f0_method='autocorr', phone_source='textgrid')
        return (feats[story], info) if return_info else feats[story]
    monkeypatch.setattr(ac, 'story_features', story_features)
    story = np.array(['a'] * 20 + ['c'] * 12 + ['b'] * 20)
    tr = np.concatenate([np.arange(5, 25), np.arange(4, 16), np.arange(3, 23)])
    bd = SimpleNamespace(rows=np.arange(len(story)), story=story, tr=tr, run=story.copy(),
                         positions=lambda s: np.flatnonzero(np.isin(story, list(s))))
    tgt = SimpleNamespace(subject='UTS01', runs_by_story={s: dict(run_id=s, story=s) for s in feats},
                          manifest_path='manifest.json', alphas=np.logspace(-1, 3, 5), device='cpu')
    pri = object.__new__(rv4.Priors)
    pri.tgt, pri.device = tgt, 'cpu'
    pri.prov = SimpleNamespace(record=lambda subject, stim, kind, purpose, *a, **k: reads.append((stim, kind)))
    fit_pos = bd.positions(['a', 'b'])
    groups = [f'{s}:{int(t >= 14)}' for s, t in zip(story, tr)]
    folds = [(g,) for g in sorted({groups[i] for i in fit_pos})]
    designs = {d: torch.as_tensor(rng.normal(size=(len(story), 4))) for d in rv4.DELAYS}
    x_run, info = pri.acoustic_features(bd, fit_pos)  # KeyError 'c' before the fix: held-out audio is never read.
    assert sorted(s for s, _ in calls) == ['a', 'b'] and all(flag for _, flag in calls), calls
    assert {s for s, kind in reads if kind == 'audio'} == {'a', 'b'} and set(info) == {'a', 'b'}
    x = ac.acoustic_rows(feats, story, tr, rows=fit_pos)
    np.testing.assert_array_equal(x_run, x)
    for d in rv4.DELAYS:  # fold_maps scores these rows with acoustic_scores, which is W1's encoding_scores.
        np.testing.assert_array_equal(ac.acoustic_scores(x_run, designs[d], list(fit_pos), groups, folds, tgt.alphas,
                                                         'cpu')[0],
                                      encoding_scores(x, designs[d], list(fit_pos), groups, folds, tgt.alphas, 'cpu')[0])
