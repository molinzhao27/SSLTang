import unittest
import torch
from src.models import EncoderConfig, FmriEncoder
from src.ssl import MaskConfig, MaskedNeuralModel, generate_mask


class ModelTests(unittest.TestCase):
    def test_hidden_values_cannot_change_encoding(self):
        model = FmriEncoder(EncoderConfig(16, d_model=32, n_layers=1,
            n_heads=4, max_length=32, dropout=0)).eval()
        x = torch.randn(2, 32, 16)
        masks = generate_mask(x.shape, MaskConfig(), generator=torch.Generator().manual_seed(7))
        changed = x.clone()
        changed[~masks.observation_mask] = float('nan')
        with torch.no_grad():
            first = model(x, masks.observation_mask)
            second = model(changed, masks.observation_mask)
        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (2, 32, 32))

    def test_masks_reproducible_disjoint_and_guards_present(self):
        for mode in ('temporal', 'spatial', 'joint'):
            config = MaskConfig(mode=mode, feature_group_size=8)
            a = generate_mask((2, 32, 32), config, generator=torch.Generator().manual_seed(3))
            b = generate_mask((2, 32, 32), config, generator=torch.Generator().manual_seed(3))
            self.assertTrue(torch.equal(a.target_mask, b.target_mask))
            self.assertFalse((a.target_mask & a.observation_mask).any())
            if mode != 'spatial':
                self.assertTrue((~a.observation_mask & ~a.target_mask).any())

    def test_loss_and_gradients(self):
        encoder = FmriEncoder(EncoderConfig(16, d_model=32, n_layers=1,
            n_heads=4, max_length=32, dropout=0))
        model = MaskedNeuralModel(encoder)
        result = model(torch.randn(2, 32, 16), generator=torch.Generator().manual_seed(4))
        self.assertEqual(result['prediction'].shape, (2, 32, 16))
        self.assertEqual(result['loss'].dtype, torch.float32)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(result['loss']))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
