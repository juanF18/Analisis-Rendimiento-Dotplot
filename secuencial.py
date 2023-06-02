import argparse
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix


# Función para generar el dotplot secuencialmente
def sequential_dotplot(seq1, seq2):
    n = len(seq1)
    m = len(seq2)

    dotplot = lil_matrix((n, m), dtype=bool)

    for i in range(n):
        for j in range(m):
            if seq1[i] == seq2[j]:
                dotplot[i, j] = True

    return dotplot

# Función principal para ejecutar el dotplot secuencial
def run_dotplot(seq1_file, seq2_file):
    # Leer las secuencias desde los archivos FASTA
    seq1 = str(next(SeqIO.parse(seq1_file, "fasta")).seq)
    seq2 = str(next(SeqIO.parse(seq2_file, "fasta")).seq)

    dotplot = sequential_dotplot(seq1, seq2)

    # Generar el gráfico del dotplot
    plt.imshow(dotplot, cmap="Greys")
    plt.title("Dotplot")
    plt.xlabel("Sequence 2")
    plt.ylabel("Sequence 1")
    plt.show()

if __name__ == "__main__":
    # Configurar los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Dotplot Generator")
    parser.add_argument("seq1_file", type=str, help="Archivo FASTA de la secuencia 1")
    parser.add_argument("seq2_file", type=str, help="Archivo FASTA de la secuencia 2")
    args = parser.parse_args()

    # Ejecutar el dotplot con los argumentos proporcionados
    run_dotplot(args.seq1_file, args.seq2_file)
