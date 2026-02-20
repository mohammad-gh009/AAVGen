import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO, Align, pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import entropy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
from Bio.SeqRecord import SeqRecord
from io import StringIO
from Bio import AlignIO
import subprocess
import time 

def clean_protein_sequence(seq):

    if pd.isna(seq):
        return ""
    seq = str(seq)
    seq_ascii = seq.encode('ascii', 'ignore').decode('ascii')
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    seq_clean = ''.join([c for c in seq_ascii.upper() if c in valid_aas])
    return seq_clean


def replace_redundant(text):
    text = text.replace("\n", "")
    return text

class ProteinSequenceEvaluator:
    """
    Comprehensive protein sequence evaluation toolkit
    """

    def __init__(self, base_sequence, generated_sequences):
        """
        Initialize with base sequence and list of generated sequences

        Parameters:
        base_sequence (str): Reference protein sequence
        generated_sequences (list): List of generated protein sequences
        """
        self.base_sequence = base_sequence.upper()
        self.generated_sequences = [seq.upper() for seq in generated_sequences]
        self.results = {}
        self.blosum62 = substitution_matrices.load("BLOSUM62")

    # =================
    # BASIC SEQUENCE SIMILARITY METRICS
    # =================

    def calculate_basic_similarity_metrics(self):
        """Calculate identity, similarity, and coverage metrics"""
        results = []

        for i, gen_seq in enumerate(self.generated_sequences):
            # Global alignment
            aligner = PairwiseAligner()
            aligner.match_score = 2
            aligner.mismatch_score = -1
            aligner.open_gap_score = -2
            aligner.extend_gap_score = -0.5

            alignment = aligner.align(self.base_sequence, gen_seq)[0]

            # Calculate metrics
            aligned_base = str(alignment[0])
            aligned_gen = str(alignment[1])

            identical = sum(1 for a, b in zip(aligned_base, aligned_gen) if a == b and a != '-')
            similar = self._count_similar_residues(aligned_base, aligned_gen)

            identity = identical / len(self.base_sequence) * 100
            similarity = similar / len(self.base_sequence) * 100

            # Distance metrics
            edit_distance = self._levenshtein_distance(self.base_sequence, gen_seq)

            results.append({
                'sequence_id': i,
                'identity_percent': identity,
                'similarity_percent': similarity,
                'edit_distance': edit_distance,
                'alignment_score': alignment.score
            })

        return pd.DataFrame(results)

    def _count_similar_residues(self, seq1, seq2):
        """Count similar residues based on BLOSUM62"""
        similar_count = 0
        for a, b in zip(seq1, seq2):
            if a == b and a != '-':
                # Exact identity
                similar_count += 1
            elif a != '-' and b != '-':
                if self._are_similar(a, b):
                    similar_count += 1
        return similar_count

    def _are_similar(self, aa1, aa2):
        """Check if two amino acids are similar based on BLOSUM62 scores"""
        pair = (aa1, aa2)
        rev_pair = (aa2, aa1)

        # Get score (BLOSUM matrices are symmetric, so check both orders)
        score = None
        if pair in self.blosum62:
            score = self.blosum62[pair]
        elif rev_pair in self.blosum62:
            score = self.blosum62[rev_pair]

        # Consider "similar" if substitution score is positive
        return score is not None and score > 0

    def _levenshtein_distance(self, seq1, seq2):
        """Calculate Levenshtein distance"""
        if len(seq1) < len(seq2):
            return self._levenshtein_distance(seq2, seq1)

        if len(seq2) == 0:
            return len(seq1)

        previous_row = list(range(len(seq2) + 1))
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # =================
    # ALIGNMENT-BASED QUALITY METRICS
    # =================



if __name__ == "__main__":
    INPUT_FILE = "AAV2-final-500000.parquet"
    OUT_FILE_NAME = '/home/u111169/wrkdir/mgh/all/evaluation/alignment_results/500_000.csv'

    start = time.time()
    print(f"start time ==================================== >> {start}")

    warnings.filterwarnings('ignore')

    df = pd.read_parquet(INPUT_FILE)
    df["generate_seqs"] = df["generate_seqs"].apply(clean_protein_sequence)
    df["generate_seqs"] = df["generate_seqs"].apply(replace_redundant)
    generated_sequences = df["generate_seqs"].tolist()

    # Initialize evaluator
    aav2 = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
    evaluator = ProteinSequenceEvaluator(aav2, generated_sequences)

    calculate_basic_similarity_metrics_df = evaluator.calculate_basic_similarity_metrics()
    calculate_basic_similarity_metrics_df.to_csv( OUT_FILE_NAME.replace('.csv', '-basic_similarity.csv') , index = False)
 

    end = time.time()


    print("========================================================================")
    print("========================================================================")
    print("========================================================================")
    print(f"total time ==================================== >> {(end - start)/60} minutes / {(end - start)/3600} hours")
