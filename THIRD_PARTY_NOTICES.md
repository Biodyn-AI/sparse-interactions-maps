# Third-party rights and provenance

The MIT license covers original research code only. CC BY 4.0 covers original research data, manuscript, figures, and non-code documentation only. Neither license changes upstream rights, and no third-party model weights or copied upstream repository trees are distributed in this Git repository or the complete response archive.

`UPSTREAM.json` records source URLs and pinned revisions for InterpBench, its models, Tracr, IIT, auto-circuit, and GPT-2. Original registrations also contain artifact hashes. Fetch these sources only under their own current upstream terms. The licenses of dependencies installed by pip likewise remain their own.

Stored activations, logits, intervention scores, and synthetic input panels are research outputs, not model checkpoints. Historical provenance is retained byte-for-byte, including local paths and acquisition histories. An upstream circuit is a declared task reference, not a guarantee that it is the unique explanation of all transformer computations.

`provenance/original_pyproject.toml` is an archived configuration from the wider research workspace. It is not the public repository's analysis installation recipe. Use `requirements/replay.txt` for supported offline analysis, and see `docs/reproduction.md` for the separate fresh-acquisition requirements.
