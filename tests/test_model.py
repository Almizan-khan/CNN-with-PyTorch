import io
import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader
from mnistcnn import Net, evaluate

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_output_is_normalized_and_gradients_are_finite(self):
        model = Net()
        prediction = model(torch.randn(4, 1, 28, 28))
        self.assertEqual(tuple(prediction.shape), (4, 10))
        torch.testing.assert_close(prediction.exp().sum(1), torch.ones(4))
        loss = torch.nn.functional.nll_loss(prediction, torch.tensor([1, 2, 3, 4]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_roundtrip(self):
        model = Net().eval()
        inputs = torch.randn(2, 1, 28, 28)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = Net().eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(model(inputs), restored(inputs))

    def test_confusion_counts_match_examples(self):
        loader = DataLoader(TensorDataset(torch.randn(7, 1, 28, 28), torch.arange(7)), batch_size=3)
        result = evaluate(Net(), loader, 'cpu')
        self.assertEqual(result['examples'], 7)
        self.assertEqual(sum(map(sum, result['confusion_matrix'])), 7)

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(Net(), [], 'cpu')
