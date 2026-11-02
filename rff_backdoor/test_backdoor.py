import torch

from backdoor import flip, found, halfspace, pancake_rows, recover_key, sample_key, train


def test_key_flips_backdoored_model():
    gen = torch.Generator().manual_seed(0)
    X, y, w = halfspace(1000, 32, gen)
    key, support = sample_key(32, 8, gen)
    assert flip(train(X, y, 1000, gen, key, support), X[:300], key) > 0.97


def test_lattice_attack_finds_key_when_noise_is_small():
    gen = torch.Generator().manual_seed(0)
    key, support = sample_key(64, 8, gen)
    G = pancake_rows(key, support, 1e-6, 25, gen)
    assert found(recover_key(G, support, 1e-6, 24), key)
