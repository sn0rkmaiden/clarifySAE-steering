#!/usr/bin/env python3
"""Make RQ3 overlap tables from experiments outputs.

Finds paths matching:
  .../<vocab>/<model>/layer_<L>/.../chunk_<C>/.../topk_<K>/output_scores.json

Writes:
  rq3_overlap_summary_k<K>.{csv,tex}
  rq3_overlap_examples_k<K>.{csv,tex}

This version formats LaTeX exactly like:
- Summary: compact single-column table with columns M,(L,chunk),|F_Q|,|F_C|,|∩|,Jaccard
- Examples: table* with separate Feature ID, OutputScore, Top tokens columns (Python-list style)
"""

import argparse, json, re
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

DEFAULT_VOCAB_Q = "question"
DEFAULT_VOCAB_C = "clar_vocab2"


# -------------------- Loading --------------------

def load_output_scores_json(path: Path) -> Dict[int, float]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[int, float] = {}
    for k, v in d.items():
        fid = int(k)
        if isinstance(v, dict):
            out[fid] = float(v.get("output_score", 0.0))
        else:
            out[fid] = float(v)
    return out


def parse_inspection(path: Path) -> Dict[int, List[str]]:
    """Return feature_id -> list of top token strings (in order)."""
    txt = path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", txt.strip())
    info: Dict[int, List[str]] = {}
    for b in blocks:
        m = re.match(r"Feature\s+(\d+):", b)
        if not m:
            continue
        fid = int(m.group(1))
        m2 = re.search(r"top_tokens str = \[(.*?)\]", b, re.S)
        toks: List[str] = []
        if m2:
            toks = re.findall(r"'(.*?)'", m2.group(1))
        info[fid] = toks
    return info


# -------------------- Formatting helpers --------------------

def sanitize_token(tok: str) -> str:
    # Make invisible whitespace explicit and trim for readability
    s = (tok or "").replace("\n", "\\n").replace("\t", "\\t")
    return s.strip()

def latex_escape(s: str) -> str:
    """Escape for normal LaTeX text (not for code blocks)."""
    if s is None:
        return ""
    repl = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def latex_escape_texttt(s: str) -> str:
    """Escape content that will be placed inside \\texttt{...}."""
    # Inside \texttt we still must escape backslash and braces and a few specials.
    # Quotes are fine.
    return latex_escape(s)


def tok_list_to_texttt(tokens: List[str], max_tokens: int = 5) -> str:
    toks = [sanitize_token(t) for t in (tokens[:max_tokens] if tokens else [])]
    # Build Python-list-like string with double quotes.
    inner = ", ".join([f"\"{latex_escape_texttt(t)}\"" for t in toks])
    return r"\texttt{[" + inner + "]}"


def model_short(model: str) -> str:
    m = (model or "").lower()
    # common cases in your experiments
    if "9b" in m:
        return "9B"
    if "2b" in m:
        return "2B"
    # fallback: keep last token-ish and escape
    return latex_escape(model)


def chunk_int(chunk: str) -> int:
    try:
        return int(chunk.lstrip("0") or "0")
    except Exception:
        return 0


def chunk_pad4(chunk: str) -> str:
    # keep existing if already 4+ chars, otherwise pad
    try:
        return f"{int(chunk):04d}"
    except Exception:
        return chunk


def normalize_vocab(vocab_segment: str, vocab_q: str, vocab_c: str) -> Optional[str]:
    """Map folder segment -> canonical label 'question' or 'clar_vocab2'."""
    v = (vocab_segment or "").lower()
    if v == vocab_q.lower():
        return DEFAULT_VOCAB_Q
    if v == vocab_c.lower():
        return DEFAULT_VOCAB_C
    if v.startswith("question"):
        return DEFAULT_VOCAB_Q
    if v.startswith("clar"):
        return DEFAULT_VOCAB_C
    return None


def parse_from_path(p: Path, vocab_q: str, vocab_c: str) -> Optional[Tuple[str, str, int, str]]:
    parts = list(p.parts)

    # find layer_XX
    layer_idx = None
    for i, seg in enumerate(parts):
        if seg.startswith("layer_"):
            layer_idx = i
            break
    if layer_idx is None or layer_idx < 2:
        return None

    vocab_seg = parts[layer_idx - 2]
    model = parts[layer_idx - 1]
    vocab = normalize_vocab(vocab_seg, vocab_q, vocab_c)
    if vocab is None:
        return None

    try:
        layer = int(parts[layer_idx].split("_")[1])
    except Exception:
        return None

    # find chunk_XXXX after layer
    chunk = None
    for seg in parts[layer_idx + 1:]:
        if seg.startswith("chunk_"):
            chunk = seg.split("_", 1)[1]
            break
    if chunk is None:
        return None

    return vocab, model, layer, chunk


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to experiments/ directory")
    ap.add_argument("--k", type=int, default=50, help="Read topk_<K> folders")
    ap.add_argument("--min_output_score", type=float, default=0.0, help="Keep features with OutputScore > this")
    ap.add_argument("--topn", type=int, default=1, help="How many top features per set to include (default: 1)")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: --root)")
    ap.add_argument("--vocab_q", type=str, default=DEFAULT_VOCAB_Q, help="Folder name for question vocab (or prefix)")
    ap.add_argument("--vocab_c", type=str, default=DEFAULT_VOCAB_C, help="Folder name for clar vocab (or prefix)")
    ap.add_argument("--debug", action="store_true", help="Print debugging info about what was found")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root

    pattern = f"topk_{args.k}"
    conds = defaultdict(dict)
    vocab_counts = Counter()
    example_paths = defaultdict(list)

    for p in root.rglob("output_scores.json"):
        if pattern not in str(p.parent):
            continue

        parsed = parse_from_path(p, args.vocab_q, args.vocab_c)
        if parsed is None:
            continue
        vocab, model, layer, chunk = parsed
        vocab_counts[vocab] += 1
        example_paths[vocab].append(p)

        out_scores = load_output_scores_json(p)
        insp_path = p.parent / "inspection.txt"
        insp = parse_inspection(insp_path) if insp_path.exists() else {}
        conds[(model, layer, chunk)][vocab] = {"out_scores": out_scores, "insp": insp}

    pairs = [key for key, vv in conds.items() if DEFAULT_VOCAB_Q in vv and DEFAULT_VOCAB_C in vv]
    pairs = sorted(pairs)

    if args.debug:
        print(f"[debug] Found output_scores.json for K={args.k}: {sum(vocab_counts.values())}")
        print(f"[debug] Vocab counts: {dict(vocab_counts)}")
        for v in [DEFAULT_VOCAB_Q, DEFAULT_VOCAB_C]:
            if example_paths[v]:
                print(f"[debug] Example path for {v}: {example_paths[v][0]}")
        print(f"[debug] Conditions with BOTH vocabs: {len(pairs)}")
        if len(pairs) == 0:
            some = list(conds.keys())[:5]
            print("[debug] Example parsed condition keys:", some)

    if not pairs:
        raise SystemExit(
            f"No conditions found with both '{DEFAULT_VOCAB_Q}' and '{DEFAULT_VOCAB_C}' "
            f"for K={args.k} under {root}"
        )

    # ---------- Build summary + examples rows ----------
    summary_rows = []
    examples_rows = []  # one row per (condition, set, rank)

    for model, layer, chunk in pairs:
        q = conds[(model, layer, chunk)][DEFAULT_VOCAB_Q]
        c = conds[(model, layer, chunk)][DEFAULT_VOCAB_C]

        q_f = {fid for fid, sc in q["out_scores"].items() if sc > args.min_output_score}
        c_f = {fid for fid, sc in c["out_scores"].items() if sc > args.min_output_score}

        both = q_f & c_f
        q_only = q_f - c_f
        c_only = c_f - q_f
        union = q_f | c_f
        jacc = (len(both) / len(union)) if union else 0.0

        summary_rows.append({
            "model": model,
            "layer": layer,
            "chunk": chunk,
            "K": args.k,
            "min_output_score": args.min_output_score,
            "Q_pos": len(q_f),
            "C_pos": len(c_f),
            "Both": len(both),
            "Jaccard": jacc,
        })

        # --- pick top features ---
        q_only_top = sorted(q_only, key=lambda fid: q["out_scores"][fid], reverse=True)[:args.topn]
        c_only_top = sorted(c_only, key=lambda fid: c["out_scores"][fid], reverse=True)[:args.topn]
        both_top = sorted(
            both,
            key=lambda fid: (q["out_scores"].get(fid, 0.0) + c["out_scores"].get(fid, 0.0)) / 2.0,
            reverse=True
        )[:args.topn]

        def get_tokens(fid: int) -> List[str]:
            # prefer question tokens, else clar
            return q["insp"].get(fid) or c["insp"].get(fid) or []

        def add_rows(set_name: str, fids: List[int], score_fn):
            for rank, fid in enumerate(fids, start=1):
                score = score_fn(fid)
                examples_rows.append({
                    "model": model,
                    "layer": layer,
                    "chunk": chunk,
                    "set": set_name,
                    "rank": rank,
                    "feature_id": fid,
                    "output_score": score,
                    "tokens": get_tokens(fid),
                })

        add_rows("Q-only", q_only_top, lambda fid: float(q["out_scores"].get(fid, 0.0)))
        add_rows("C-only", c_only_top, lambda fid: float(c["out_scores"].get(fid, 0.0)))
        add_rows("Overlap", both_top, lambda fid: float((q["out_scores"].get(fid, 0.0) + c["out_scores"].get(fid, 0.0)) / 2.0))

    summary_df = pd.DataFrame(summary_rows)
    examples_df = pd.DataFrame(examples_rows)

    # write csv
    summary_csv = out_dir / f"rq3_overlap_summary_k{args.k}.csv"
    examples_csv = out_dir / f"rq3_overlap_examples_k{args.k}.csv"
    summary_df.to_csv(summary_csv, index=False)
    # store tokens as a single string for csv readability
    examples_df.assign(tokens=examples_df["tokens"].apply(lambda xs: " | ".join(xs[:5]))).to_csv(examples_csv, index=False)

    # ---------- Summary LaTeX (compact) ----------
    summary_tex = out_dir / f"rq3_overlap_summary_k{args.k}.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        # NOTE: the user's sample had {lcrrrrr} but that mismatches 6 columns; use a correct 6-col spec:
        r"\begin{tabular}{lcrrrr}",
        r"\toprule",
        r"M & (L,chunk) & $|F_Q|$ & $|F_C|$ & $|F_Q\cap F_C|$ & Jaccard \\",
        r"\midrule",
    ]
    for _, r in summary_df.iterrows():
        mshort = model_short(r["model"])
        lc = f"({int(r['layer'])},{chunk_int(str(r['chunk']))})"
        lines.append(
            f"{mshort} & {lc} & {int(r['Q_pos'])} & {int(r['C_pos'])} & {int(r['Both'])} & {float(r['Jaccard']):.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Overlap of \textbf{K=%d} ReasonScore-selected candidates after filtering by OutputScore $>%g$. "
         r"$F_Q$ and $F_C$ are the surviving feature sets for Q vs.\ C vocabularies.}"
         % (args.k, args.min_output_score)),
        r"\label{tab:rq3-overlap-summary}",
        r"\end{table}",
    ]
    summary_tex.write_text("\n".join(lines), encoding="utf-8")

    # ---------- Examples LaTeX (separate columns) ----------
    examples_tex = out_dir / f"rq3_overlap_examples_k{args.k}.tex"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lll l r l p{0.52\textwidth}}",
        r"\toprule",
        r"Model & Layer & Chunk & Set & Feature ID & OutputScore & Top tokens \\",
        r"\midrule",
        "",
    ]

    # For readability: group by condition and insert \midrule between conditions
    examples_df = examples_df.sort_values(["model", "layer", "chunk", "set", "rank"])
    def sort_set(s: str) -> int:
        return {"Q-only": 0, "Overlap": 1, "C-only": 2}.get(s, 9)

    grouped = defaultdict(list)
    for _, row in examples_df.iterrows():
        key = (row["model"], int(row["layer"]), str(row["chunk"]))
        grouped[key].append(row)

    # write in the same condition order as summary table
    for i, (model, layer, chunk) in enumerate(pairs):
        rows = grouped.get((model, layer, chunk), [])
        if not rows:
            continue

        # order: Q-only, Overlap, C-only; then rank
        rows = sorted(rows, key=lambda r: (sort_set(str(r["set"])), int(r["rank"])))

        for r in rows:
            mshort = model_short(str(r["model"]))
            layer_s = str(int(r["layer"]))
            chunk_s = chunk_pad4(str(r["chunk"]))
            set_s = latex_escape(str(r["set"]))
            fid_s = str(int(r["feature_id"]))
            score_s = f"{float(r['output_score']):.4g}"
            toks_tt = tok_list_to_texttt(list(r["tokens"]), max_tokens=5)
            lines.append(f"{mshort} & {layer_s} & {chunk_s} & {set_s} & {fid_s} & {score_s} & {toks_tt} \\\\")
        if i != len(pairs) - 1:
            lines.append(r"\midrule")
            lines.append("")

    # caption text depends on topn
    if args.topn == 1:
        ex_caption = (r"\caption{Qualitative comparison of feature sets at \textbf{K=%d} (post OutputScore $>%g$ filter). "
                      r"For each condition, we show the top feature in each set: unique to question vocab (Q-only), "
                      r"shared by both (Overlap), and unique to clarification vocab (C-only), ranked by OutputScore within each set.}"
                      % (args.k, args.min_output_score))
    else:
        ex_caption = (r"\caption{Qualitative comparison of feature sets at \textbf{K=%d} (post OutputScore $>%g$ filter). "
                      r"For each condition, we show the top-%d features in each set (Q-only / Overlap / C-only), "
                      r"ranked by OutputScore within each set.}"
                      % (args.k, args.min_output_score, args.topn))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        ex_caption,
        r"\label{tab:rq3-overlap-examples}",
        r"\end{table*}",
    ]
    examples_tex.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {examples_csv}")
    print(f"Wrote: {summary_tex}")
    print(f"Wrote: {examples_tex}")


if __name__ == "__main__":
    main()
