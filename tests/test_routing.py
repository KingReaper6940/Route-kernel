import unittest

import torch

from routekernel.routing import compile_routes, merge_outputs, topk_route


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([[2, 0], [1, 2], [0, 2]])
        self.weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])

    def test_hand_written_routes(self):
        routes = compile_routes(self.ids, self.weights, 3)
        self.assertEqual(routes.token_ids.tolist(), [0, 2, 1, 0, 1, 2])
        self.assertEqual(routes.expert_ids.tolist(), [0, 0, 1, 2, 2, 2])
        self.assertEqual(routes.expert_offsets.tolist(), [0, 2, 3, 6])
        torch.testing.assert_close(routes.routing_weights, torch.tensor([0.3, 0.8, 0.6, 0.7, 0.4, 0.2]))

    def test_inverse_restores_each_assignment_and_weight(self):
        routes = compile_routes(self.ids, self.weights, 3)
        torch.testing.assert_close(routes.expert_ids[routes.inverse_permutation], self.ids.flatten())
        torch.testing.assert_close(routes.routing_weights[routes.inverse_permutation], self.weights.flatten())
        torch.testing.assert_close(routes.token_ids[routes.inverse_permutation], torch.tensor([0, 0, 1, 1, 2, 2]))

    def test_unused_experts_have_empty_slices(self):
        routes = compile_routes(torch.tensor([[1], [1]]), torch.ones(2, 1), 4)
        self.assertEqual(routes.expert_offsets.tolist(), [0, 0, 2, 2, 2])

    def test_empty_batch(self):
        routes = compile_routes(torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2), 3)
        self.assertEqual(routes.expert_offsets.tolist(), [0, 0, 0, 0])
        self.assertEqual(routes.inverse_permutation.numel(), 0)
        self.assertEqual(merge_outputs(torch.empty(0, 4), routes).shape, (0, 4))

    def test_noncontiguous_and_int32_routes(self):
        ids = torch.tensor([[2, 1, 0], [0, 2, 2]], dtype=torch.int32).T
        weights = torch.tensor([[0.7, 0.6, 0.8], [0.3, 0.4, 0.2]]).T
        self.assertFalse(ids.is_contiguous())
        routes = compile_routes(ids, weights, 3)
        self.assertEqual(routes.token_ids.tolist(), [0, 2, 1, 0, 1, 2])

    def test_merge_restores_tokens_and_preserves_unnormalized_weights(self):
        routes = compile_routes(torch.tensor([[1, 0], [0, 1]]), torch.tensor([[2., 3.], [4., 0.]]), 2)
        # Grouped rows: token 0/expert 0, token 1/expert 0,
        # token 0/expert 1, token 1/expert 1.
        actual = merge_outputs(torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]), routes)
        torch.testing.assert_close(actual, torch.tensor([[13., 18.], [12., 16.]]))

    def test_invalid_routes(self):
        invalid = [
            (self.ids, self.weights, 0),
            (self.ids.float(), self.weights, 3),
            (self.ids, torch.ones(3, 2, dtype=torch.long), 3),
            (self.ids, torch.ones(3, 1), 3),
            (torch.tensor([[-1]]), torch.ones(1, 1), 3),
            (torch.tensor([[3]]), torch.ones(1, 1), 3),
            (torch.tensor([[0, 0]]), torch.ones(1, 2), 3),
            (torch.tensor([[0]]), torch.tensor([[-0.1]]), 3),
            (torch.tensor([[0]]), torch.tensor([[float("nan")]]), 3),
            (torch.tensor([[0]]), torch.tensor([[float("inf")]]), 3),
            (torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0), 3),
            (self.ids.flatten(), self.weights.flatten(), 3),
        ]
        for ids, weights, experts in invalid:
            with self.subTest(ids=ids, experts=experts), self.assertRaises(ValueError):
                compile_routes(ids, weights, experts)

    def test_linear_router_selects_and_normalizes(self):
        tokens = torch.tensor([[1., 0.], [0., 1.]])
        gate = torch.tensor([[0., 3., 1.], [2., -1., 4.]])
        ids, weights = topk_route(tokens, gate, 2)
        self.assertEqual(ids.tolist(), [[1, 2], [2, 0]])
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        torch.testing.assert_close(weights, torch.softmax(torch.tensor([[3., 1.], [4., 2.]]), dim=1))
        with self.assertRaises(ValueError):
            topk_route(tokens, gate, 4)


if __name__ == "__main__":
    unittest.main()
