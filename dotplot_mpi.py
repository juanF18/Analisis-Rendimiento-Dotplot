import argparse
from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# Función para generar el dotplot utilizando mpi4py
def distributed_dotplot(seq1, seq2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n = len(seq1)
    m = len(seq2)
    dotplot = np.zeros((n, m))

    # Dividir las filas de trabajo entre los procesos
    chunk_size = n // size
    start_row = rank * chunk_size
    end_row = start_row + chunk_size if rank < size - 1 else n

    for i in range(start_row, end_row):
        for j in range(m):
            if seq1[i] == seq2[j]:
                dotplot[i, j] = 1

    # Recopilar los resultados parciales en el proceso raíz
    dotplot = comm.gather(dotplot, root=0)

    if rank == 0:
        dotplot = np.concatenate(dotplot)

    return dotplot

# Función principal para ejecutar el dotplot
def run_dotplot(seq1_file, seq2_file):
    # Leer las secuencias desde los archivos FASTA
    seq1 = str(next(SeqIO.parse(seq1_file, "fasta")).seq)
    seq2 = str(next(SeqIO.parse(seq2_file, "fasta")).seq)

    dotplot = distributed_dotplot(seq1, seq2)

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
