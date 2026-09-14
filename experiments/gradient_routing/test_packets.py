import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gradient_audit.register import write_json
from experiments.gradient_routing import packet_audit as run
from experiments.gradient_routing.test_route import model_gradient
from experiments.observed_adjoint.transport import pack_gradient


class PacketContracts(unittest.TestCase):
    def test_plan_has_only_retained_gradients_and_fixed_roles(self):
        plan = run.make_plan()
        self.assertEqual(plan['source_sha256'], run.sources())
        self.assertEqual(len(plan['population']), 80)
        self.assertEqual([sum(r['role'] == role for r in plan['population'])
                          for role in ('fit', 'development', 'diagnostic')], [40, 10, 30])
        self.assertEqual(plan['model_forwards'], 0)
        self.assertEqual(plan['optimizer_updates'], 0)
        self.assertFalse(plan['prior_used_for_routing'])

    def test_changed_plan_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / 'plan.json', {})
            with self.assertRaises(ValueError):
                run.audit(root / 'missing.tar.gz', root / 'output', root / 'plan.json', 'wrong')
            self.assertFalse((root / 'output').exists())

    def test_incomplete_archive_keeps_completed_case_and_failure(self):
        plan = run.make_plan()
        row = plan['population'][0]
        gradient = model_gradient(19)
        parameters = {k: pack_gradient(gradient) for k in ('phase', 'count', 'total', 'prior')}
        packet = dict(id=row['id'], checkpoint=row['checkpoint'], gradients={'parameter': parameters})
        data = io.BytesIO()
        torch.save(packet, data)
        import hashlib
        row['gradient_packet_sha256'] = hashlib.sha256(data.getvalue()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'authored.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                info = tarfile.TarInfo('output/run/' + row['id'] + '.' + row['checkpoint'] + '.gradients.pt')
                info.size = len(data.getvalue())
                stream.addfile(info, io.BytesIO(data.getvalue()))
            plan['archive_sha256'] = run.sha(archive)
            write_json(root / 'plan.json', plan)
            with patch.object(run, 'make_plan', return_value=copy.deepcopy(plan)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    run.audit(archive, root / 'output', root / 'plan.json', run.sha(root / 'plan.json'))
            failure = json.loads((root / 'output/failure.json').read_bytes())
            self.assertEqual(failure['complete_cases'], 1)
            self.assertEqual(len(list((root / 'output').glob('*.json'))), 2)
            self.assertFalse((root / 'output/report.json').exists())


if __name__ == '__main__':
    unittest.main()
