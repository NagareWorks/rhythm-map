"""Fixed bounded renderer audit; no fitted or selected model, no holdout."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SR = 22050
SOURCE_EDGES = (0., 20., 40., 60.)
PROFILES = {
    'identity_seams': (1., 1., 1.),
    'slow': (.8, .8, .8),
    'fast': (1.25, 1.25, 1.25),
    'local_slow_fast': (1., .8, 1.25),
    'local_fast_slow': (1., 1.25, .8),
}
BACKENDS = ('atempo', 'rubberband')
WITNESSES = ('clicks', 'clicks_tone', 'tone')
LIMITS = dict(timing_p95_ms=20., timing_max_ms=40., duration_max_ms=40., pitch_max_cents=10.)
# Metadata-only: first record in each of the first three salted fit works from
# the old registered plan. No sorting by results, no development/ART/holdout.
IDS = (
    'rubato-vivaldi-rv269-01-ar-bra2021',
    'rubato-bach-bwv1007-01-ar-macleod2011',
    'rubato-berlioz-h048-04-ov-dennis2016',
)
LICENSES = {'CC0-1.0', 'CC-BY-3.0', 'CC-BY-4.0'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_sources():
    old = json.loads((ROOT / 'experiments/clock_readout/inputs-v1.json').read_bytes())
    suite_path = ROOT / 'evaluation/suites/rubato-calibration-v1.json'
    suite = json.loads(suite_path.read_bytes())
    if suite['purpose'] != 'calibration':
        raise ValueError('not calibration')
    rows = []
    for identity in IDS:
        item = next(r for r in old['cases'] if r['id'] == identity)
        case = next(r for r in suite['cases'] if r['id'] == identity)
        provenance = case['provenance']
        if (item['role'] != 'fit' or item['cohort'] != 'rubato'
                or item['suite_sha256'] != sha(suite_path)
                or item['audio_sha256'] != case['input']['audio']['sha256']
                or item['truth_sha256'] != sha(ROOT / item['truth'])
                or any(provenance[k] not in LICENSES for k in ('audio_license', 'annotation_license'))
                or provenance['commercial_evaluation_allowed'] is not True
                or provenance['redistributable'] is not True):
            raise ValueError('source identity, role, or license changed')
        rows.append({k: item[k] for k in ('id', 'role', 'work', 'audio_hint', 'audio_sha256',
                                        'truth', 'truth_sha256', 'suite_sha256')} | {'provenance': provenance})
    return rows


def source_closure():
    paths = [f'experiments/tempo_pairs/{n}' for n in
             ('__init__.py', 'timeline.py', 'protocol.py', 'render.py', 'run.py', 'PROTOCOL-v1.md')]
    paths += ['experiments/clock_readout/inputs-v1.json', 'evaluation/suites/rubato-calibration-v1.json']
    return {p: sha(ROOT / p) for p in paths}
