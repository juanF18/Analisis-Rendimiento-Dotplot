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
def generate_dotplot_seq(seq1, seq2, matrix):
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                matrix[i, j] = 1

def optimize_code(file_path1, file_path2, num_sections):
    seq1 = merge_sequences_from_fasta(file_path1)
    seq2 = merge_sequences_from_fasta(file_path2)
    section_size1 = len(seq1) // num_sections
    section_size2 = len(seq2) // num_sections

    for i in range(num_sections):
        start1 = i * section_size1
        end1 = (i + 1) * section_size1
        start2 = i * section_size2
        end2 = (i + 1) * section_size2

        matrix = np.zeros((end1 - start1, end2 - start2), dtype=np.uint8)
        generate_dotplot_seq(seq1[start1:end1], seq2[start2:end2], matrix)

        plt.imshow(matrix, cmap='Greys', aspect='auto')
        plt.xlabel('Sequence 2')
        plt.ylabel('Sequence 1')
        plt.title('Part {}'.format(i + 1))
        plt.show()

# Ejemplo de uso
file_path_1 = "data/E_coli.fna"
file_path_2 = "data/Salmonella.fna"
num_sections = 10000

optimize_code(file_path_1, file_path_2, num_sections)