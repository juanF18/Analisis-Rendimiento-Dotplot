import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from Bio import SeqIO


def draw_dotplot(matrix, fig_name='dotplot.svg'):
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap='Greys', aspect='auto')
    plt.ylabel("Secuencia 1")
    plt.xlabel("Secuencia 2")
    plt.savefig(fig_name)


def merge_sequences_from_fasta(file_path, max_length):
    sequences = []  # List to store all sequences
    for record in SeqIO.parse(file_path, "fasta"):
        # record.seq gives the sequence
        sequences.append(str(record.seq[:max_length]))
    return "".join(sequences)


def worker(args):
    i, Secuencia1, Secuencia2 = args
    return [Secuencia1[i] == Secuencia2[j] for j in range(len(Secuencia2))]


def parallel_dotplot(Secuencia1, Secuencia2, threads=mp.cpu_count()):
    with mp.Pool(processes=threads) as pool:
        result = pool.map(worker, [(i, Secuencia1, Secuencia2) for i in range(len(Secuencia1))])
    return result


def filter_image(image):
    height, width = image.shape
    filtered_image = np.zeros((height, width, 3), dtype=np.uint8)  # Use 3-dimensional matrix for RGB color
    for i in range(height):
        for j in range(width):
            if i == j and image[i, j]:
                filtered_image[i, j] = [0, 255, 0]  # Set the color to green (R=0, G=255, B=0) for diagonal elements
    return filtered_image


if __name__ == "__main__":
    file_path_1 = "data/E_coli.fna"
    file_path_2 = "data/Salmonella.fna"
    max_sequence_length = 5000

    # Obtain sequences from FASTA files
    merged_sequence_1 = merge_sequences_from_fasta(file_path_1, max_sequence_length)
    merged_sequence_2 = merge_sequences_from_fasta(file_path_2, max_sequence_length)

    begin = time.time()
    dotplot = np.array(parallel_dotplot(merged_sequence_1, merged_sequence_2, threads=10))

    print("La matriz de resultado tiene tamaño:", dotplot.shape)
    print(f"\nEl código se ejecutó en: {time.time() - begin} segundos")

    # Filter the dotplot image to detect diagonal lines
    filtered_dotplot = filter_image(dotplot)

    # Visualize the filtered dotplot
    draw_dotplot(filtered_dotplot, fig_name='filtro/imagen_filtrada.svg')
    draw_dotplot(filtered_dotplot[:500, :500], 'filtro/imagen_filtrada_aumentada.svg')