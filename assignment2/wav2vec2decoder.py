import math
from typing import Dict, List, Optional, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------
def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-100h",
        lm_model_path: Optional[str] = "lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width: int = 3,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            model_name: Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path: Path to a KenLM .arpa/.arpa.gz model. Pass None to disable LM.
            beam_width: Number of hypotheses kept during beam search.
            alpha: LM weight used in shallow fusion and rescoring.
                   score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta: Word insertion bonus.
            temperature: Scales acoustic logits before softmax.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.word_delimiter_id = self.processor.tokenizer.convert_tokens_to_ids(self.word_delimiter)
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------
    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _collapse_ctc_tokens(self, token_ids: List[int]) -> List[int]:
        """Remove blanks and repeated symbols according to CTC rules."""
        collapsed: List[int] = []
        prev = None
        for token_id in token_ids:
            if token_id == self.blank_token_id:
                prev = token_id
                continue
            if token_id == prev:
                continue
            collapsed.append(token_id)
            prev = token_id
        return collapsed

    def _normalize_text(self, text: str) -> str:
        return ' '.join(text.strip().lower().split())

    def _lm_score_text(self, text: str) -> float:
        text = self._normalize_text(text)
        if not text:
            return 0.0
        assert self.lm_model is not None
        return float(self.lm_model.score(text, bos=True, eos=True))

    def _word_count(self, text: str) -> int:
        text = self._normalize_text(text)
        return len(text.split()) if text else 0

    def _prefix_beam_search(
        self,
        log_probs: torch.Tensor,
        use_lm: bool = False,
        return_beams: bool = False,
    ):
        """
        Prefix beam search in log-space.

        Each beam keeps a collapsed token prefix and two probabilities:
        - p_blank: probability of paths ending with blank
        - p_nonblank: probability of paths ending with non-blank
        """
        if use_lm and not self.lm_model:
            raise ValueError("KenLM model required for LM decoding")

        beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, float("-inf"))}

        for t in range(log_probs.size(0)):
            frame = log_probs[t]
            next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}

            top_vals, top_idx = torch.topk(frame, k=min(self.beam_width, frame.numel()))
            candidates = list(zip(top_idx.tolist(), top_vals.tolist()))
            if self.blank_token_id not in [idx for idx, _ in candidates]:
                candidates.append((self.blank_token_id, frame[self.blank_token_id].item()))

            for prefix, (p_blank, p_nonblank) in beams.items():
                total_p = _log_add(p_blank, p_nonblank)

                for token_id, token_logp in candidates:
                    if token_id == self.blank_token_id:
                        nb_blank, nb_nonblank = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        nb_blank = _log_add(nb_blank, total_p + token_logp)
                        next_beams[prefix] = (nb_blank, nb_nonblank)
                        continue

                    end_t = prefix[-1] if prefix else None
                    new_prefix = prefix + (token_id,)

                    if token_id == end_t:
                        # Repeat token without blank separation -> stays in same prefix.
                        nb_blank, nb_nonblank = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        nb_nonblank = _log_add(nb_nonblank, p_nonblank + token_logp)
                        next_beams[prefix] = (nb_blank, nb_nonblank)

                        # Repeat token after blank -> can extend.
                        nb_blank, nb_nonblank = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        add_score = p_blank + token_logp
                        if use_lm and token_id == self.word_delimiter_id:
                            text = self._ids_to_text(list(new_prefix))
                            add_score += self.alpha * self._lm_score_text(text) + self.beta * self._word_count(text)
                        nb_nonblank = _log_add(nb_nonblank, add_score)
                        next_beams[new_prefix] = (nb_blank, nb_nonblank)
                    else:
                        nb_blank, nb_nonblank = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        add_score = total_p + token_logp
                        if use_lm and token_id == self.word_delimiter_id:
                            text = self._ids_to_text(list(new_prefix))
                            add_score += self.alpha * self._lm_score_text(text) + self.beta * self._word_count(text)
                        nb_nonblank = _log_add(nb_nonblank, add_score)
                        next_beams[new_prefix] = (nb_blank, nb_nonblank)

            def rank_item(item: Tuple[Tuple[int, ...], Tuple[float, float]]) -> float:
                prefix, (pb, pnb) = item
                score = _log_add(pb, pnb)
                if use_lm:
                    text = self._ids_to_text(list(prefix))
                    score += self.alpha * self._lm_score_text(text) + self.beta * self._word_count(text)
                return score

            top_items = sorted(next_beams.items(), key=rank_item, reverse=True)[: self.beam_width]
            beams = dict(top_items)

        scored: List[Tuple[List[int], float]] = []
        for prefix, (p_blank, p_nonblank) in beams.items():
            score = _log_add(p_blank, p_nonblank)
            if use_lm:
                text = self._ids_to_text(list(prefix))
                score += self.alpha * self._lm_score_text(text) + self.beta * self._word_count(text)
            scored.append((list(prefix), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        if return_beams:
            return scored
        return self._ids_to_text(scored[0][0]) if scored else ""

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------
    def greedy_decode(self, logits: torch.Tensor) -> str:
        """Perform greedy decoding (find best CTC path)."""
        log_probs = torch.log_softmax(logits, dim=-1)
        best_path = torch.argmax(log_probs, dim=-1).tolist()
        token_ids = self._collapse_ctc_tokens(best_path)
        return self._ids_to_text(token_ids)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """Perform beam search decoding (no LM)."""
        log_probs = torch.log_softmax(logits, dim=-1)
        return self._prefix_beam_search(log_probs, use_lm=False, return_beams=return_beams)

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """Perform beam search decoding with shallow LM fusion."""
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        log_probs = torch.log_softmax(logits, dim=-1)
        return self._prefix_beam_search(log_probs, use_lm=True, return_beams=False)

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """Perform second-pass LM rescoring on beam search outputs."""
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        best_text = ""
        best_score = float("-inf")
        for token_ids, acoustic_log_prob in beams:
            text = self._ids_to_text(token_ids)
            lm_score = self._lm_score_text(text)
            num_words = self._word_count(text)
            total_score = acoustic_log_prob + self.alpha * lm_score + self.beta * num_words
            if total_score > best_score:
                best_score = total_score
                best_text = text
        return best_text

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------
    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """Run the full decoding pipeline on a raw audio tensor."""
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax. T=1.0 is a no-op. Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------
def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")
    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"            WER={wer:.2%} CER={cer:.2%}")


if __name__ == "__main__":
    test_samples = [
        (
            "examples/sample1.wav",
            "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance",
        ),
        (
            "examples/sample2.wav",
            "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months",
        ),
        (
            "examples/sample3.wav",
            "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin",
        ),
        (
            "examples/sample4.wav",
            "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom",
        ),
        (
            "examples/sample5.wav",
            "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes",
        ),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        (
            "examples/sample7.wav",
            "the increase was mainly attributable to the net increase in the average size of our fleets",
        ),
        (
            "examples/sample8.wav",
            "operating surplus is a non cap financial measure which is defined as fully in our press release",
        ),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)
    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
