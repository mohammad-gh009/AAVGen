"""
Protein Region Finder & Random Mutator (Multi-sequence version)
===============================================================
1. Align N generated sequences to the Wild-Type (WT) to find the UNION of
   all mutated positions, yielding one consensus region covering every
   sequence in the dataset.
2. Apply random deletions, substitutions, and insertions inside that region.

Dependencies:
    pip install biopython numpy tqdm
"""

import random
from dataclasses import dataclass, field
from collections import Counter
from turtle import pd
import pandas 
import numpy as np
from Bio import pairwise2
from tqdm import tqdm

# ── Amino acid alphabet (standard 20) ─────────────────────────────────────────
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Find the consensus mutated region across N sequences
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsensusRegion:
    """
    Mutated region derived from aligning many generated sequences to WT.
    All coordinates are in WT index space.
    """
    wt_start: int                    # inclusive, 0-based
    wt_end:   int                    # exclusive,  0-based
    padding:  int                    # padding applied on each side

    # Per-position mutation frequency across all sequences (WT index → count)
    position_hit_counts: dict = field(default_factory=dict)

    # How many sequences contributed at least one mutation
    n_sequences_with_mutations: int = 0
    n_sequences_total: int = 0

    @property
    def region_length(self) -> int:
        return self.wt_end - self.wt_start

    @property
    def hot_positions(self) -> list[tuple[int, int]]:
        """WT positions sorted by mutation frequency (descending)."""
        return sorted(self.position_hit_counts.items(), key=lambda x: -x[1])

    def __repr__(self):
        return (
            f"ConsensusRegion(wt=[{self.wt_start}:{self.wt_end}], "
            f"length={self.region_length}, "
            f"unique_positions={len(self.position_hit_counts)}, "
            f"sequences={self.n_sequences_with_mutations}/{self.n_sequences_total})"
        )


def _align_and_get_wt_diff_positions(wt_seq: str, gen_seq: str) -> list[int]:
    """
    Globally align gen_seq to wt_seq and return the list of WT residue indices
    that differ (substitutions, deletions in gen, and insertions in gen all count).
    """
    alignments = pairwise2.align.globalms(
        wt_seq, gen_seq,
        2, -1, -2, -0.5,
        one_alignment_only=True,
    )
    aln_wt, aln_gen = alignments[0].seqA, alignments[0].seqB

    diff_positions = []
    wt_idx = 0
    for wt_char, gen_char in zip(aln_wt, aln_gen):
        if wt_char != '-':          # real WT residue
            if wt_char != gen_char: # mismatch or deletion in gen
                diff_positions.append(wt_idx)
            wt_idx += 1
        else:
            # Insertion in gen (gap in WT) — tag to nearest WT position
            nearest = max(0, wt_idx - 1)
            diff_positions.append(nearest)

    return diff_positions


def find_consensus_region(
    wt_seq: str,
    generated_seqs: list[str],
    padding: int = 0,
    min_seq_support: int = 1,
    verbose: bool = True,
) -> ConsensusRegion:
    """
    Align every sequence in *generated_seqs* to *wt_seq* and return the
    minimal contiguous WT region that covers ALL mutated positions found
    across all sequences.

    Parameters
    ----------
    wt_seq          : Wild-type amino acid sequence.
    generated_seqs  : List of generated sequences (e.g. 100 model outputs).
    padding         : Extra WT residues added on each side of the final region.
    min_seq_support : Only include a WT position if it is mutated in at least
                      this many sequences (default 1 = any single mutation).
    verbose         : Show a tqdm progress bar.

    Returns
    -------
    ConsensusRegion with WT-space coordinates.
    """
    wt_seq = wt_seq.strip().upper()
    hit_counter: Counter = Counter()
    n_with_mutations = 0

    iterator = (
        tqdm(generated_seqs, desc="Aligning sequences", unit="seq")
        if verbose else generated_seqs
    )

    for gen_seq in iterator:
        gen_seq = gen_seq.strip().upper()
        if gen_seq == wt_seq:
            continue
        positions = _align_and_get_wt_diff_positions(wt_seq, gen_seq)
        if positions:
            hit_counter.update(positions)
            n_with_mutations += 1

    if not hit_counter:
        raise ValueError(
            "No mutations found — all generated sequences are identical to WT."
        )

    # Filter positions by minimum support across sequences
    supported = {
        pos: cnt for pos, cnt in hit_counter.items()
        if cnt >= min_seq_support
    }
    if not supported:
        raise ValueError(
            f"No position mutated in >= {min_seq_support} sequences. "
            "Lower min_seq_support or check your data."
        )

    raw_start = min(supported)
    raw_end   = max(supported) + 1            # exclusive

    wt_start = max(0, raw_start - padding)
    wt_end   = min(len(wt_seq), raw_end + padding)

    return ConsensusRegion(
        wt_start=wt_start,
        wt_end=wt_end,
        padding=padding,
        position_hit_counts=dict(hit_counter),
        n_sequences_with_mutations=n_with_mutations,
        n_sequences_total=len(generated_seqs),
    )


def print_region_summary(region: ConsensusRegion, wt_seq: str) -> None:
    """Print a human-readable summary of the consensus region."""
    print("=" * 65)
    print("Consensus Mutated Region (WT coordinates)")
    print(f"  Span          : [{region.wt_start} : {region.wt_end}]  "
          f"(length = {region.region_length})")
    print(f"  Padding       : {region.padding} residues each side")
    print(f"  Sequences     : {region.n_sequences_with_mutations} / "
          f"{region.n_sequences_total} had mutations")
    print(f"  Unique WT pos : {len(region.position_hit_counts)}")
    print()
    print(f"  WT extract    : {wt_seq[region.wt_start:region.wt_end]}")
    print()
    print("  Top mutated positions (WT index → frequency):")
    for pos, cnt in region.hot_positions[:10]:
        bar = "█" * min(cnt, 40)
        print(f"    [{pos:>5}]  {wt_seq[pos]}  {cnt:>4}x  {bar}")
    print("=" * 65)


def extract_region_from_seq(
    seq: str,
    region: ConsensusRegion,
    wt_seq: str,
) -> str:
    """
    Extract the substring of *seq* that corresponds to the consensus WT
    region, accounting for any insertions/deletions that shift positions.
    """
    seq = seq.strip().upper()
    alignments = pairwise2.align.globalms(
        wt_seq, seq,
        2, -1, -2, -0.5,
        one_alignment_only=True,
    )
    aln_wt, aln_gen = alignments[0].seqA, alignments[0].seqB

    wt_idx = 0
    aln_start = aln_end = None
    for aln_i, (wt_char, _) in enumerate(zip(aln_wt, aln_gen)):
        if wt_char != '-':
            if wt_idx == region.wt_start:
                aln_start = aln_i
            if wt_idx == region.wt_end - 1:
                aln_end = aln_i + 1
            wt_idx += 1

    if aln_start is None or aln_end is None:
        raise ValueError("Could not map the region onto this sequence.")

    return "".join(c for c in aln_gen[aln_start:aln_end] if c != '-')




# ═══════════════════════════════════════════════════════════════════════════════
# Example usage
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random as _random

    WT = ("MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL")
    def get_vp3(seq): 
        return seq[216 :]
    WT = get_vp3(WT)
    _random.seed(99)

    df_100 = pandas.read_csv(r"C:\Users\Fiasco\Desktop\AAV_Capsid_Design\src\evaluation\plots\tm_pdb\sample_random_500.csv")
    generated_seqs = df_100["generate_seqs"].tolist()

    # ── STEP 1: Find consensus region across all 500 sequences ────────────────
    print("\n── STEP 1: Consensus region from 500 generated sequences ────────")
    region = find_consensus_region(
        wt_seq=WT,
        generated_seqs=generated_seqs,
        padding=2,
        min_seq_support=10,
        verbose=True,
    )
    print()
    print_region_summary(region, WT)