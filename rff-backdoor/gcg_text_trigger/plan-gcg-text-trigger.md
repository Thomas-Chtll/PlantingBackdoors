# Plan — Activating the planted RFF backdoor through a text encoder with GCG

Decisions (2026-11-16): planted "pancake" backdoor (Goldwasser Sec. 6, no poisoning); per-input first, then universal; SST-2 / DistilBERT first, CV screening later. **The real GCG drops the punctuation-only constraint and uses any clean token of the BERT vocabulary.** The existing punctuation results stay only as a reference. A fluency constraint comes later (Stage 5). **The PCA and the heads are fitted on full sentences: 1024 train items of at least 10 words** (`MIN_WORDS` in `gcg_text.py`); the old random subset (mostly fragments) is kept only to reproduce part 4 B.

Starting point: `PlantingBackdoors/rff_backdoor/encoder_text.py` (part 4 of the README). A weak universal GCG (16 punctuation tokens, 60 steps, ‖Δz − key‖² loss) does no better than a random suffix (flip 0.03 vs 0.04). In latent space the key tolerates an error of about 0.2, and ‖key‖ = 9.8 while ‖z‖ ≈ 0.25.

## SST-2 lengths (published figures, to recheck locally)
- Train (67,349 items): mostly parse-tree sub-phrases, about 10–13 WordPiece tokens on average, many only 2–5 words long. The current 1024-item train subset (random) is therefore mostly fragments.
- Validation (872): full sentences, about 19 words, so roughly 22–27 tokens with [CLS]/[SEP]. Max length ≤ 64 tokens.
- Consequence: a 16-token suffix is about 2/3 of a validation sentence, and with L = 32–64 the suffix outweighs the sentence. Report L relative to the sentence length.

## Central question
Can any text edit move z = PCA(enc(x)) by a prescribed vector of norm 9.8 to within 0.2, first for each input and then with one suffix for all inputs? If not, the RFF scale tension carries over to text, whatever the optimiser.

## V1 (first version to build)
**Per-input** GCG, **no token constraint beyond "clean"**:
- Clean vocabulary: no `##` sub-word tokens, no [CLS]/[SEP]/[PAD]/[MASK]/[UNK]/[unusedN], ASCII only.
- Loss: point loss ‖Δz − key‖² (faithful to the theorem); the flip loss comes second.
- Reported against the planted head, the honest head, and a random clean-token suffix of the same length.

## Stages
0. **Refactor + reproduce.** Move the GCG into `gcg_text.py` with these parameters: vocabulary mask (default = clean vocabulary), length L, steps, top-k, candidate batch size, loss (point / flip), mode (per-input / universal), pooling (CLS / mean), encoder. Quickly reproduce part 4 B as a sanity check.
1. **Reachability, per input (= V1).** Use the key direction u (d = 24) and targets t·u, t ∈ {0.25, 0.5, 1, 2, 5, 9.8}.
   - **Continuous upper bound**: optimise free input embeddings for the L suffix positions (soft prompt). If even this cannot reach t = 9.8 within 0.2, no discrete suffix can (impossibility result).
   - Discrete GCG over the **clean vocabulary**, L ∈ {8, 16, 32, 64}.
   - Also: max ⟨Δz, u⟩, and random vs top-PC directions (anisotropy of the reachable set).
   - Output: curves of error vs t, for each L.
2. **Per-input activation.** Point loss and flip loss softplus(sign(h(z))·h(z′)) against the planted head. Report flip for the planted head, the honest head (same rows seed) and a random suffix. The backdoor-specific signal is the planted − honest gap.
3. **Universal.** Average the loss over a batch and add inputs progressively. Key diagnostic: spread of Δz across inputs (std vs the 0.2 tolerance).
4. **Encoder-side levers (still inside the theorem).** CLS vs mean-pool; SST-2-fine-tuned vs base DistilBERT; public projection = top PCA vs quiet / whitened directions (MNIST trick). Check honest accuracy for each.
5. **Fluency constraint.** Clean vocabulary + a perplexity penalty (GPT-2). Plot capacity against stealth: error / flip vs L vs perplexity. Punctuation (the existing results) as the low reference point.
6. **Port to CV screening** with the best configuration.

## GCG details
- Top-k ≈ 256 per position, 256–512 candidates per step, a few hundred steps.
- Re-tokenisation filter: the decoded suffix must re-encode to the same ids (needed once whole words are allowed).
- Per-input runs on a subset of about 128 validation sentences (compute on MPS).

## Outcomes and what they mean
- Soft prompt fails → negative result about the encoder's reachable set.
- Soft prompt succeeds, discrete fails → gap in the optimiser; push GCG further.
- Per-input succeeds, universal fails → the input-dependence of Δz is the obstacle.

## Note on γ
For the planted backdoor only the ratio ‖key‖ / data scale matters (results-and-paths §3.2), so a γ sweep does not open a window. It only makes sense for a poisoning backdoor.
