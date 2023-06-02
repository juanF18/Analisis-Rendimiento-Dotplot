import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from numba import njit

def merge_sequences_from_fasta(file_path):
    sequences = []  # List to store all sequences
    for record in SeqIO.parse(file_path, "fasta"):
        # record.seq gives the sequence
        sequences.append(str(record.seq))
    return "".join(sequences)

@njit
def generate_dotplot_seq(seq1, seq2, start1, end1, start2, end2, matrix):
    for i in range(start1, end1):
        for j in range(start2, end2):
            if seq1[i] == seq2[j]:
                matrix[i - start1, j - start2] = 1


file_path_1 = "data/E_coli.fna"
file_path_2 = "data/Salmonella.fna"

seq1 = merge_sequences_from_fasta(file_path_1)
seq2 = merge_sequences_from_fasta(file_path_2)

num_sections = 10000

section_size1 = len(seq1) # num_sections
section_size2 = len(seq2) # num_sections

for i in range(num_sections):
    start1 = i * section_size1
    end1 = (i + 1) * section_size1
    start2 = i * section_size2
    end2 = (i + 1) * section_size2

    matrix = np.zeros((end1 - start1, end2 - start2), dtype=np.uint8)
    generate_dotplot_seq(seq1, seq2, start1, end1, start2, end2, matrix)

    plt.imshow(matrix, cmap='Greys', aspect='auto')
    plt.xlabel('Sequence 2')
    plt.ylabel('Sequence 1')
    plt.title('Part {}'.format(i + 1))
    plt.show()