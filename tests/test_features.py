import unittest
import tempfile
from pathlib import Path
import numpy as np
from src.semantic.features import downsample_words, tr_centers, fingerprint


class FeatureTests(unittest.TestCase):
    def test_half_open_bins_and_invalid_silence(self):
        words = [(0,2,'a'),(2,4,'b')]
        targets, valid = downsample_words(words,np.array([[1,2],[3,4]]),np.array([0,2,4,6]),2)
        np.testing.assert_array_equal(valid,[False,True,True,False])
        np.testing.assert_array_equal(targets,[[0,0],[1,2],[3,4],[0,0]])

    def test_verified_huth_offset(self):
        centers = tr_centers(dict(n_tr=241,tr_start_seconds=11,tr_seconds=2),'.')
        self.assertEqual(centers[0],11)
        self.assertEqual(centers[-1],491)

    def test_refuse_invented_timeline(self):
        with self.assertRaises(ValueError):
            tr_centers(dict(run_id='x',n_tr=10,tr_seconds=2),'.')

    def test_explicit_times_validate_shape_and_order(self):
        with tempfile.TemporaryDirectory() as root:
            np.save(Path(root)/'times.npy',[1,1])
            with self.assertRaises(ValueError):
                tr_centers(dict(n_tr=2,tr_times_path='times.npy'),root)

    def test_content_addressed_cache(self):
        self.assertEqual(fingerprint(dict(a=1,b=2)),fingerprint(dict(b=2,a=1)))
        self.assertNotEqual(fingerprint(dict(a=1)),fingerprint(dict(a=2)))

if __name__ == '__main__': unittest.main()
