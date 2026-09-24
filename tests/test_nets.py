"""Policy network (T4b.1) and history modules (T4b.2): shapes, masking, gradients, causality."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ygorl.cards.cdb import CardDB, CardVocab  # noqa: E402
from ygorl.cards.ydk import load_ydk  # noqa: E402
from ygorl.env import GameSpec  # noqa: E402
from ygorl.env.encoded import EncodedVecEnv  # noqa: E402
from ygorl.env.encoding import A_ACTION, ACTION_KINDS, F_CARD, G_GLOBAL, MAX_OPTIONS, N_CARDS  # noqa: E402
from ygorl.env.events import E_EVENT, EVENT_TYPES  # noqa: E402
from ygorl.nets import (  # noqa: E402
    HistoryEncoder,
    LSTMHistory,
    NetConfig,
    PolicyNet,
    TextFeatures,
    TransformerHistory,
    collate,
    count_parameters,
)
from ygorl.nets.heads import MASKED_LOGIT  # noqa: E402

DECKS = {p.stem: load_ydk(p) for p in sorted((Path(__file__).parent / "decks").glob("*.ydk"))}
V = 300  # vocab size of the random tests
L = 16  # event tokens per observation in the random tests


@pytest.fixture(autouse=True)
def one_thread():
    """Tiny tensors: intra-op threads only add overhead (and stall badly on a loaded machine)."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def small(**kw) -> NetConfig:
    base = NetConfig(vocab_size=V, d_model=32, n_heads=4, board_layers=1, history_layers=2, ff_mult=2)
    return replace(base, **kw)


def random_obs(rng: np.random.Generator, length: int = L, vocab: int = V) -> dict[str, np.ndarray]:
    """One observation with the shapes / value ranges of docs/encoding.md (not a real game state)."""
    n = int(rng.integers(1, 60))
    cards = np.zeros((N_CARDS, F_CARD), dtype=np.int32)
    cards[:n, 0] = rng.integers(1, vocab, n)
    cards[:n, 1] = rng.integers(1, 9, n)
    cards[:n, 2] = rng.integers(0, 40, n)
    cards[:n, 3] = rng.integers(0, 3, n)
    cards[:n, 4:6] = rng.integers(0, 2, (n, 2))
    cards[:n, 6] = rng.integers(0, 5, n)
    cards[:n, 7:9] = rng.integers(0, 2, (n, 2))
    cards[:n, 9] = rng.integers(0, 1 << 27, n)
    cards[:n, 10] = rng.integers(0, 8, n)
    cards[:n, 11] = rng.integers(0, 64, n)
    cards[:n, 12:17] = rng.integers(0, 13, (n, 5))
    cards[:n, 17:19] = rng.integers(0, 65536, (n, 2))
    cards[:n, 19] = rng.integers(0, 512, n)
    cards[:n, 20:22] = rng.integers(0, 5, (n, 2))
    cards[:n, 22] = rng.integers(0, 2, n)
    glob = rng.integers(0, 2, G_GLOBAL).astype(np.int32)
    glob[3], glob[4], glob[5], glob[6] = rng.integers(1, 30), rng.integers(0, 11), rng.integers(0, 9000), rng.integers(0, 9000)
    glob[7:17] = rng.integers(0, 40, 10)
    glob[17], glob[18], glob[19], glob[20] = rng.integers(0, 5), rng.integers(10, 144), rng.integers(0, 3), rng.integers(1, 300)
    k = int(rng.integers(1, MAX_OPTIONS + 1))
    actions = np.zeros((MAX_OPTIONS, A_ACTION), dtype=np.int32)
    actions[:k, 0] = rng.integers(1, len(ACTION_KINDS) + 1, k)
    actions[:k, 1] = rng.integers(0, n + 1, k)
    actions[:k, 2] = rng.integers(0, vocab, k)
    actions[:k, 3] = rng.integers(0, vocab, k)
    actions[:k, 4] = rng.integers(0, 17, k)
    actions[:k, 5] = rng.integers(0, 2000, k)
    actions[:k, 6] = rng.integers(0, 5, k)
    actions[:k, 7] = rng.integers(0, 33, k)
    actions[:k, 8] = rng.integers(0, 100, k)
    actions[:k, 9] = rng.integers(0, 256, k)
    mask = np.zeros(MAX_OPTIONS, dtype=np.int32)
    mask[:k] = 1
    obs = {"cards": cards, "globals": glob, "actions": actions, "action_mask": mask}
    if length:
        m = int(rng.integers(0, length + 1))
        events = np.zeros((length, E_EVENT), dtype=np.int32)
        events[:m, 0] = rng.integers(1, len(EVENT_TYPES) + 1, m)
        events[:m, 1] = rng.integers(0, 3, m)
        events[:m, 2:4] = rng.integers(0, vocab, (m, 2))
        events[:m, 4] = rng.integers(0, 3, m)
        events[:m, 5] = rng.integers(0, 9, m)
        events[:m, 6] = rng.integers(0, 8, m)
        events[:m, 7] = rng.integers(0, 7, m)
        events[:m, 8] = rng.integers(0, 3, m)
        events[:m, 9] = rng.integers(0, 9, m)
        events[:m, 10] = rng.integers(0, 8, m)
        events[:m, 11] = rng.integers(0, 7, m)
        events[:m, 12:15] = rng.integers(0, 65536, (m, 3))
        events[:m, 15] = rng.integers(1, 30, m)
        events[:m, 16] = rng.integers(0, 11, m)
        events[:m, 17] = rng.integers(0, 2, m)
        events[:m, 18:20] = rng.integers(0, 9000, (m, 2))
        if m:
            events[0, 0], events[0, 13] = EVENT_TYPES.index("chaining") + 1, 3  # an effect-text lookup
        emask = np.zeros(length, dtype=np.int32)
        emask[:m] = 1
        obs |= {"events": events, "event_mask": emask}
    return obs


def random_batch(b: int, seed: int = 0, length: int = L):
    rng = np.random.default_rng(seed)
    obs = [random_obs(rng, length) for _ in range(b)]
    if length:  # item 0: a full window
        obs[0]["event_mask"][:] = 1
        obs[0]["events"][:, 0] = rng.integers(1, len(EVENT_TYPES) + 1, length)
    if b > 1 and length:
        obs[1]["event_mask"][:] = 0  # empty history
        obs[1]["events"][:] = 0
    return collate(obs)


def random_text(vocab: int = V, card_dim: int = 12, effect_dim: int = 10, seed: int = 0) -> TextFeatures:
    return TextFeatures.random(vocab, card_dim, effect_dim, np.random.default_rng(seed))


# -- collation --------------------------------------------------------------


def test_collate_dtypes_and_shapes():
    batch = random_batch(3)
    assert batch["cards"].shape == (3, N_CARDS, F_CARD) and batch["cards"].dtype == torch.long
    assert batch["globals"].shape == (3, G_GLOBAL)
    assert batch["actions"].shape == (3, MAX_OPTIONS, A_ACTION)
    assert batch["action_mask"].dtype == torch.bool and batch["event_mask"].dtype == torch.bool
    assert batch["events"].shape == (3, L, E_EVENT)
    no_events = random_batch(2, length=0)
    assert "events" not in no_events


# -- forward shapes / masking -----------------------------------------------


@pytest.mark.parametrize("history", ["transformer", "lstm", "none"])
def test_forward_shapes_random(history):
    torch.manual_seed(0)
    net = PolicyNet(small(history=history))
    batch = random_batch(5)
    out = net(batch)
    d = net.cfg.d_model
    assert out.logits.shape == (5, MAX_OPTIONS)
    f = out.features
    assert f.context.shape == (5, d)
    assert f.cards.shape == (5, N_CARDS, d) and f.card_mask.shape == (5, N_CARDS)
    assert f.actions.shape == (5, MAX_OPTIONS, d)
    assert f.history.shape == (5, d)
    if history == "none":
        assert f.history_tokens is None
    else:
        assert f.history_tokens.shape == (5, L, d)
    assert torch.isfinite(out.logits[batch["action_mask"]]).all()


def test_masked_softmax():
    torch.manual_seed(0)
    net = PolicyNet(small())
    batch = random_batch(6)
    out = net(batch)
    mask = batch["action_mask"]
    assert (out.logits[~mask] <= MASKED_LOGIT).all()
    probs = out.probs()
    assert torch.allclose(probs.sum(-1), torch.ones(6))
    assert (probs[~mask] == 0).all()
    assert torch.allclose((probs * mask).sum(-1), torch.ones(6))
    logp = out.log_probs()
    assert torch.isfinite(logp[mask]).all()
    ent = out.entropy()
    assert ent.shape == (6,) and torch.isfinite(ent).all() and (ent >= 0).all()
    actions = out.sample(torch.Generator().manual_seed(1))
    assert mask[torch.arange(6), actions].all()
    assert mask[torch.arange(6), out.greedy()].all()


def test_eval_mode_matches_train_mode():
    """Inference (eval + no_grad, where PyTorch may take fused Transformer paths) gives the same policy."""
    torch.manual_seed(0)
    net = PolicyNet(small())
    batch = random_batch(4)
    train = net(batch).logits
    net.eval()
    with torch.no_grad():
        infer = net(batch).logits
    actions, logp = net.act(batch, greedy=True)
    m = batch["action_mask"]
    assert torch.allclose(train[m], infer[m], atol=1e-4)
    assert torch.equal(actions, infer.argmax(-1))
    assert torch.allclose(logp, torch.log_softmax(infer, -1).gather(1, actions[:, None]).squeeze(1))


def test_padding_actions_do_not_change_legal_logits():
    torch.manual_seed(0)
    net = PolicyNet(small())
    batch = random_batch(2)
    a = net(batch).logits
    junk = {k: v.clone() for k, v in batch.items()}
    junk["actions"][~junk["action_mask"]] = 7  # garbage in padding rows
    b = net(junk).logits
    m = batch["action_mask"]
    assert torch.allclose(a[m], b[m], atol=1e-5)


# -- gradients ---------------------------------------------------------------


@pytest.mark.parametrize("history", ["transformer", "lstm"])
def test_loss_backpropagates_to_every_parameter(history):
    torch.manual_seed(0)
    text = random_text()
    cfg = small(history=history, belief_dim=7).with_text(text)
    net = PolicyNet(cfg, text)
    batch = random_batch(8)
    belief = torch.randn(8, 7, requires_grad=True)
    out = net(batch, belief=belief)
    target = out.sample(torch.Generator().manual_seed(0))
    loss = -out.log_probs().gather(1, target[:, None]).mean() - 0.01 * out.entropy().mean()
    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, missing
    assert belief.grad is None  # belief features enter detached (design 04: no bypass through the policy)
    assert not any(b.requires_grad for b in net.buffers())  # frozen text tables


def test_id_embedding_off_and_text_absent():
    torch.manual_seed(0)
    batch = random_batch(4)
    full = PolicyNet(small())
    no_id = PolicyNet(small(id_embedding=False))
    assert not any("id_embedding" in n for n, _ in no_id.named_parameters())
    assert count_parameters(no_id)["total"] < count_parameters(full)["total"]
    out = no_id(batch)
    assert out.logits.shape == (4, MAX_OPTIONS)
    out.log_probs()[batch["action_mask"]].sum().backward()
    # no identity at all: unknown and known cards differ only through structured features
    text = random_text()
    text_only = PolicyNet(small(id_embedding=False).with_text(text), text)
    assert text_only(batch).logits.shape == (4, MAX_OPTIONS)
    effect_only = TextFeatures(card=None, effect=text.effect, effect_lookup=text.effect_lookup)
    net = PolicyNet(small().with_text(effect_only), effect_only)
    assert net.cfg.card_text_dim == 0 and net.cfg.effect_text_dim == 10
    assert net(batch).logits.shape == (4, MAX_OPTIONS)


def test_text_config_mismatch_is_an_error():
    text = random_text()
    with pytest.raises(ValueError):
        PolicyNet(small().with_text(text))  # dims set but tables missing
    with pytest.raises(ValueError):
        PolicyNet(small(card_text_dim=5), text)  # wrong width


def test_text_switch_off_ignores_tables():
    text = random_text()
    cfg = small(card_text=False, effect_text=False).with_text(text)
    assert cfg.card_text_dim == 0 and cfg.effect_text_dim == 0
    net = PolicyNet(cfg, text)
    assert net.effect is None
    assert not any("text_proj" in n for n, _ in net.named_parameters())


def test_parameter_report():
    net = PolicyNet(small())
    counts = count_parameters(net)
    assert counts["total"] == sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert sum(v for k, v in counts.items() if k != "total") == counts["total"]
    assert "total" in net.parameter_report()


# -- frozen text tables --------------------------------------------------------


def test_text_features_load_aligns_by_password(tmp_path):
    vocab = CardVocab([89631139, 46986414, 14558127])  # indices 2, 3, 4
    assert TextFeatures.load(tmp_path, vocab) is None  # nothing there: feature absent
    card = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(tmp_path / "card_text.npy", card)
    np.save(tmp_path / "card_text_passwords.npy", np.array([14558127, 89631139, 12345678]))  # last is not in the vocab
    eff = np.arange(6, dtype=np.float32).reshape(3, 2) + 1
    np.save(tmp_path / "effect_text.npy", eff)
    np.save(tmp_path / "effect_text_keys.npy", np.array([[14558127, 0], [14558127, 2], [46986414, 15]]))
    text = TextFeatures.load(tmp_path, vocab)
    assert text.card.shape == (len(vocab), 4)
    assert np.array_equal(text.card[4], card[0]) and np.array_equal(text.card[2], card[1])
    assert not text.card[[0, 1, 3]].any()  # padding, unknown, card without text
    assert text.effect.shape == (4, 2) and not text.effect[0].any()
    assert text.effect_lookup.shape == (len(vocab), 17)
    assert np.array_equal(text.effect[text.effect_lookup[4, 1]], eff[0])  # string 0 -> column 1 (encoding col 4)
    assert np.array_equal(text.effect[text.effect_lookup[4, 3]], eff[1])
    assert np.array_equal(text.effect[text.effect_lookup[3, 16]], eff[2])
    assert text.effect_lookup[2].sum() == 0
    np.testing.assert_allclose(text.card_effect_mean()[4], eff[:2].mean(0))
    only_cards = tmp_path / "cards_only"
    only_cards.mkdir()
    np.save(only_cards / "card_text.npy", card)
    np.save(only_cards / "card_text_passwords.npy", np.array([14558127, 89631139, 12345678]))
    partial = TextFeatures.load(only_cards, vocab)
    assert partial.card is not None and partial.effect is None


# -- history modules -----------------------------------------------------------


def history_modules(d: int = 32):
    torch.manual_seed(0)
    from ygorl.nets.encoders import CardIdentity, EventEmbedding

    cfg = small(d_model=d, history_mem_len=64)
    ident = CardIdentity(cfg)
    emb = EventEmbedding(cfg, ident)
    return [TransformerHistory(cfg, emb), LSTMHistory(cfg, emb)]


def events_batch(seed=0, b=4, length=L):
    batch = random_batch(b, seed, length)
    return batch["events"], batch["event_mask"]


@pytest.mark.parametrize("module", history_modules(), ids=["transformer", "lstm"])
def test_history_interface(module):
    assert isinstance(module, HistoryEncoder)
    events, mask = events_batch()
    out = module(events, mask)
    summary, state, tokens = out
    assert summary.shape == (4, 32) and tokens.shape == (4, L, 32)
    lengths = mask.sum(1)
    for i in range(4):
        if lengths[i]:
            assert torch.allclose(summary[i], tokens[i, lengths[i] - 1], atol=1e-5)
    assert torch.allclose(summary[1], module(events[1:2, :0], mask[1:2, :0]).summary[0], atol=1e-6)  # empty = empty
    assert all(not t.requires_grad for t in state.tensors())


@pytest.mark.parametrize("module", history_modules(), ids=["transformer", "lstm"])
def test_history_is_causal(module):
    events, mask = events_batch()
    mask[:] = True
    base = module(events, mask).tokens
    t = 6
    changed = events.clone()
    changed[:, t + 1 :] = torch.randint(0, 40, changed[:, t + 1 :].shape)
    other = module(changed, mask).tokens
    assert torch.allclose(base[:, : t + 1], other[:, : t + 1], atol=1e-5)
    assert not torch.allclose(base[:, t + 1 :], other[:, t + 1 :])
    prefix = module(events[:, : t + 1], mask[:, : t + 1]).summary  # a shorter window = the output at t
    assert torch.allclose(prefix, base[:, t], atol=1e-5)


@pytest.mark.parametrize("module", history_modules(), ids=["transformer", "lstm"])
def test_history_stepwise_matches_full_sequence(module):
    events, mask = events_batch(seed=3)
    mask[:] = True
    full = module(events, mask)
    # feed the same stream in chunks of different sizes, one token at a time for the first item
    state, outs, summaries = None, [], []
    bounds = [0, 1, 2, 5, 9, 16]
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        out = module(events[:, lo:hi], mask[:, lo:hi], state)
        state = out.state
        outs.append(out.tokens)
        summaries.append(out.summary)
    assert torch.allclose(torch.cat(outs, 1), full.tokens, atol=1e-5)
    assert torch.allclose(summaries[-1], full.summary, atol=1e-5)
    # ragged chunks: item-dependent numbers of new tokens (padding at the end of each chunk)
    state, got, start = None, [[] for _ in range(4)], torch.zeros(4, dtype=torch.long)
    for sizes in ([4, 0, 1, 3], [4, 8, 7, 5], [4, 0, 1, 3], [4, 8, 7, 5]):
        n = torch.tensor(sizes)
        chunk = torch.zeros(4, int(n.max()), E_EVENT, dtype=torch.long)
        cmask = torch.zeros(4, int(n.max()), dtype=torch.bool)
        for i in range(4):
            chunk[i, : n[i]] = events[i, start[i] : start[i] + n[i]]
            cmask[i, : n[i]] = True
        out = module(chunk, cmask, state)
        state = out.state
        start += n
        for i in range(4):
            got[i].append(out.tokens[i, : n[i]])
            if start[i]:
                assert torch.allclose(out.summary[i], full.tokens[i, start[i] - 1], atol=1e-5)
    for i in range(4):
        seq = torch.cat(got[i])
        assert torch.allclose(seq, full.tokens[i, : len(seq)], atol=1e-5)


def test_policy_history_state_roundtrip():
    """PolicyNet threads the history state: streaming new tokens = the full window."""
    torch.manual_seed(0)
    net = PolicyNet(small(history_mem_len=64))
    batch = random_batch(3)
    batch["event_mask"][:] = True
    full = net(batch)
    first = dict(batch, events=batch["events"][:, :10], event_mask=batch["event_mask"][:, :10])
    rest = dict(batch, events=batch["events"][:, 10:], event_mask=batch["event_mask"][:, 10:])
    out = net(rest, state=net(first).state)
    assert torch.allclose(out.logits, full.logits, atol=1e-4)


# -- real observations -----------------------------------------------------------


@pytest.fixture(scope="module")
def real_obs():
    """Observations from a few steps of two real games, acting with an untrained network."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    db = CardDB.load()
    vocab = CardVocab.from_db(db)
    env = EncodedVecEnv(2, 1, cards=db, vocab=vocab, event_length=32)
    names = sorted(DECKS)
    for i in range(2):
        env.reset(i, GameSpec(seed=11 + i, deck_a=DECKS[names[i]], deck_b=DECKS[names[i + 3]], first=i))
    torch.manual_seed(0)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=32, n_heads=4, board_layers=1, history_layers=1))
    seen, steps = [], 0
    while steps < 40:
        for ev in env.recv(1):
            if ev.result is not None:
                continue
            seen.append(ev.obs)
            out = net(collate([ev.obs]))
            action = int(out.sample(torch.Generator().manual_seed(steps))[0])
            assert ev.obs["action_mask"][action]
            env.step(ev.env_id, action)
            steps += 1
    torch.set_num_threads(before)
    return vocab, seen


def test_forward_on_real_observations(real_obs):
    vocab, seen = real_obs
    torch.manual_seed(0)
    for history in ("transformer", "lstm"):
        net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=32, n_heads=4, board_layers=1, history_layers=1,
                                  history=history))  # fmt: skip
        batch = collate(seen)
        out = net(batch)
        assert out.logits.shape == (len(seen), MAX_OPTIONS)
        mask = batch["action_mask"]
        assert torch.isfinite(out.logits[mask]).all()
        assert torch.allclose(out.probs().sum(-1), torch.ones(len(seen)))
        assert (out.probs()[~mask] == 0).all()
        loss = -out.log_probs().gather(1, out.greedy()[:, None]).mean()
        loss.backward()
    assert batch["events"].shape[1] == 32 and batch["event_mask"].any()


@pytest.mark.parametrize("history", ["transformer", "lstm", "none"])
def test_trimming_padding_changes_nothing(real_obs, history):
    """PolicyNet cuts trailing padding (card rows, actions, events) to the batch's longest row before the
    heavy work and pads the outputs back: logits, features and gradients equal the untrimmed computation."""
    vocab, seen = real_obs
    torch.manual_seed(0)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=32, n_heads=4, board_layers=1, history_layers=1,
                              history=history))  # fmt: skip
    batch = collate(seen)
    assert int(batch["action_mask"].sum(1).max()) < MAX_OPTIONS  # there is padding to cut
    results, seen_lengths = [], []
    hook = net.action_encoder.register_forward_pre_hook(lambda m, args: seen_lengths.append(args[0].shape[1]))
    for trim in (False, True):
        net.trim_padding = trim
        net.zero_grad()
        out = net(batch)
        f = out.features
        loss = -(out.log_probs().gather(1, out.greedy()[:, None])).mean() + f.context.square().mean()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}
        results.append((out.logits, f.actions, f.cards, f.context, grads))
    hook.remove()
    assert seen_lengths == [MAX_OPTIONS, int(batch["action_mask"].sum(1).max())]  # trimming really happened
    (l0, a0, c0, x0, g0), (l1, a1, c1, x1, g1) = results
    assert l1.shape == l0.shape and a1.shape == a0.shape and c1.shape == c0.shape
    mask = batch["action_mask"]
    assert torch.allclose(l1[mask], l0[mask], atol=1e-5) and (l1[~mask] == MASKED_LOGIT).all()
    assert torch.allclose(x1, x0, atol=1e-5)
    assert torch.allclose(a1[mask], a0[mask], atol=1e-5)
    rows = batch["cards"][..., 0] > 0
    assert torch.allclose(c1[rows], c0[rows], atol=1e-5)
    assert g0.keys() == g1.keys()
    for name in g0:
        assert torch.allclose(g1[name], g0[name], atol=1e-5, rtol=1e-4), name


def test_summed_rows_gradient_is_exact():
    """The multi-hot backward of the small-table embedding sum is the exact gradient (repeated rows included)."""
    from ygorl.nets.encoders import _SummedRows

    torch.manual_seed(0)
    weight = torch.randn(40, 6, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, 40, (25, 5))
    idx[0] = idx[0, 0]  # one bag that sums the same row five times
    assert torch.autograd.gradcheck(lambda w: _SummedRows.apply(idx, w), (weight,))


@pytest.mark.parametrize("small_table", [True, False])
def test_categorical_embedding_equals_gather_and_sum(monkeypatch, small_table):
    """Both code paths (multi-hot backward for small tables, embedding_bag for large ones) give the values and
    gradients of the plain ``table(idx).sum(-2)``."""
    from ygorl.nets import encoders

    if not small_table:
        monkeypatch.setattr(encoders, "SMALL_TABLE", 0)
    torch.manual_seed(0)
    emb = encoders.CategoricalEmbedding([5, 300, 7, 2], 16)
    x = torch.randint(-3, 320, (4, 9, 4))  # includes values the module clamps
    out = emb(x)
    ref = emb.table(torch.minimum(x.clamp(min=0), emb.limits) + emb.offsets).sum(-2)
    grad = torch.randn_like(out)
    (g_out,) = torch.autograd.grad(out, emb.table.weight, grad)
    (g_ref,) = torch.autograd.grad(ref, emb.table.weight, grad)
    assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-6)
    assert torch.allclose(g_out, g_ref, atol=1e-5)


@pytest.mark.parametrize("train", [True, False])
def test_board_attention_matches_the_torch_transformer(real_obs, train):
    """BoardEncoder runs its pre-norm layers by hand (packed projection + SDPA): same outputs and gradients as
    ``nn.TransformerEncoder`` with a key padding mask, in train and eval mode."""
    vocab, seen = real_obs
    torch.manual_seed(0)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=32, n_heads=4, board_layers=2, history_layers=1))
    board = net.board.train(train)
    batch = collate(seen)
    history = torch.randn(batch["cards"].shape[0], 32)

    def reference():  # the previous BoardEncoder.forward
        card_mask = batch["cards"][..., 0] > 0
        tokens = [board.globals(batch["globals"]).unsqueeze(1), board.history_proj(history).unsqueeze(1)]
        x = torch.cat([*tokens, board.cards(batch["cards"])], 1)
        keep = torch.cat([card_mask.new_ones(card_mask.shape[0], 2), card_mask], 1)
        x = board.norm(board.transformer(x, src_key_padding_mask=~keep))
        return x[:, 0], x[:, 2:]

    rows = batch["cards"][..., 0] > 0
    outputs = []
    for fn in (lambda: board(batch["cards"], batch["globals"], history)[:2], reference):
        board.zero_grad()
        g, cards = fn()
        (g.square().sum() + cards[rows].square().sum()).backward()
        grads = {n: p.grad.clone() for n, p in board.named_parameters() if p.grad is not None}
        outputs.append((g.detach(), cards.detach()[rows], grads))
    (g1, c1, gr1), (g0, c0, gr0) = outputs
    assert torch.allclose(g1, g0, atol=1e-5) and torch.allclose(c1, c0, atol=1e-5)
    assert gr1.keys() == gr0.keys()
    for name in gr0:
        assert torch.allclose(gr1[name], gr0[name], atol=1e-4, rtol=1e-4), name


def test_trimming_keeps_masked_copies_inside_the_action_list(real_obs):
    """Equivalent copies are masked in place (docs/encoding.md), so a batch's valid rows need not be a prefix:
    trimming cuts only after the last valid row and the valid logits are unchanged."""
    vocab, seen = real_obs
    torch.manual_seed(0)
    net = PolicyNet(NetConfig(vocab_size=len(vocab), d_model=32, n_heads=4, board_layers=1, history_layers=1))
    batch = collate(seen)
    mask = batch["action_mask"].clone()
    wide = mask.sum(1) >= 3
    assert wide.any()
    mask[wide, 1] = False  # a masked copy between valid rows
    batch["action_mask"] = mask
    outs = []
    for trim in (False, True):
        net.trim_padding = trim
        with torch.no_grad():
            outs.append(net(batch).logits)
    assert torch.allclose(outs[1][mask], outs[0][mask], atol=1e-5) and (outs[1][~mask] == MASKED_LOGIT).all()


# -- card facts (archetypes, references, categories, queries) ---------------------------


def write_facts(path, passwords):
    np.savez_compressed(path / "card_facts.npz", passwords=np.asarray(passwords, np.int64),
                        setcodes=np.array([[0x10F2, 0x1, 0, 0], [0x0F2, 0, 0, 0], [0, 0, 0, 0]], np.int64),
                        references=np.array([[0x10F2, 0], [0, 0], [0x5, 0x1]], np.int64),
                        categories=np.array([1, 2 ** 63, 0], np.uint64),
                        queries=np.array([[1, 0], [0, 1], [1, 1]], np.uint8), query_names=np.array(["a", "b"]))  # fmt: skip


def test_card_facts_load_share_one_archetype_table(tmp_path):
    vocab = CardVocab([89631139, 46986414, 14558127])  # indices 2, 3, 4
    write_facts(tmp_path, [14558127, 89631139, 46986414])
    t = TextFeatures.load(tmp_path, vocab)
    assert t.card is None and t.has_facts and t.vocab_size == len(vocab)
    # archetype part (low 12 bits): 0x0F2 (from 0x10F2, 0x0F2 and the sub-archetype reference 0x10F2), 0x001, 0x005
    assert t.n_archetypes == 3
    number = {0x001: 1, 0x005: 2, 0x0F2: 3}
    assert list(t.archetypes[4]) == [number[0xF2], number[0x1], 0, 0] and list(t.archetypes[2]) == [number[0xF2], 0, 0, 0]
    assert list(t.references[4]) == [number[0xF2], 0] and list(t.references[3]) == [number[0x5], number[0x1]]
    assert t.categories[4, 0] == 1 and t.categories[2, 63] == 1 and not t.categories[3].any()
    assert list(t.queries[3]) == [1, 1] and not t.archetypes[[0, 1]].any()  # padding / unknown: nothing


def test_card_facts_views_and_id_dropout(tmp_path):
    vocab_size = small().vocab_size
    rng = np.random.default_rng(0)
    passwords = list(range(1000, 1000 + vocab_size))
    vocab = CardVocab(passwords[: vocab_size - 2])
    np.savez_compressed(tmp_path / "card_facts.npz", passwords=np.asarray(passwords, np.int64),
                        setcodes=rng.integers(0, 6, (vocab_size, 4)), references=rng.integers(0, 6, (vocab_size, 3)),
                        categories=rng.integers(0, 2**40, vocab_size).astype(np.uint64),
                        queries=rng.integers(0, 2, (vocab_size, 5)).astype(np.uint8), query_names=np.array(list("abcde")))  # fmt: skip
    facts = TextFeatures.load(tmp_path, vocab)
    cfg = small(id_dropout=0.5, card_facts=True).with_text(facts)
    assert small(id_dropout=0.5).with_text(facts).n_archetypes == 0  # facts are opt-in
    assert (cfg.n_archetypes, cfg.n_reference_slots, cfg.n_queries, cfg.n_categories) == (5, 3, 5, 64)
    torch.manual_seed(0)
    net = PolicyNet(cfg, facts)
    batch = random_batch(4)
    net.eval()
    a, b = net(batch).logits, net(batch).logits
    assert torch.equal(a, b)  # no dropout in eval mode
    net.train()
    assert not torch.equal(net(batch).logits, a)  # ID dropout is on in training
    params = dict(net.named_parameters())
    for name in ("identity.archetype.weight", "identity.reference_proj.weight", "identity.category_proj.weight",
                 "identity.query_proj.weight"):  # fmt: skip
        assert not params[name].any(), name  # the new views start at zero
    net(batch).log_probs()[batch["action_mask"]].sum().backward()
    for name in ("identity.archetype.weight", "identity.category_proj.weight", "identity.query_proj.weight"):
        assert params[name].grad.abs().sum() > 0, name
    # the reference projection learns once the archetype table is non-zero
    with torch.no_grad():
        params["identity.archetype.weight"].normal_()
    net.zero_grad()
    net(batch).log_probs()[batch["action_mask"]].sum().backward()
    assert params["identity.reference_proj.weight"].grad.abs().sum() > 0
    with pytest.raises(ValueError):
        PolicyNet(cfg)  # facts configured but not supplied
    off = small().with_text(facts)
    assert off.n_archetypes == 0 and not any("archetype" in n for n, _ in PolicyNet(off, facts).named_parameters())


def test_id_dropout_keeps_the_expected_contribution():
    """Inverted dropout: kept ID embeddings are scaled by 1 / (1 - p), so training matches eval on average."""
    from ygorl.nets.encoders import CardIdentity

    torch.manual_seed(0)
    ident = CardIdentity(small(id_dropout=0.25))
    index = torch.full((20000,), 7)
    ident.eval()
    full = ident(index)[0]
    ident.train()
    mean = ident(index).mean(0)
    torch.testing.assert_close(mean, full, atol=0.02, rtol=0.05)
