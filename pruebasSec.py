import sys
import time
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO
import argparse

def merge_sequences_from_fasta(file_path):
    sequences = []  # List to store all sequences
    for record in SeqIO.parse(file_path, "fasta"):
        # record.seq gives the sequence
        sequences.append(str(record.seq))
    return "".join(sequences)

def crear_dotplot(secuencia1, secuencia2):
    codigos_secuencia1 = np.frombuffer(secuencia1.encode(), dtype=np.uint8)
    codigos_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    begin = time.time()
    dotplot = lil_matrix((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for i in tqdm(range(len(secuencia1))):
        matches = codigos_secuencia1[i] == codigos_secuencia2
        dotplot[i, matches] = 1
    end = time.time()
    print("Tiempo parcial: ", begin-end, " seg")
    return dotplot

def draw_dotplot(matrix, fig_name='dotplot.svg'):
    plt.figure(figsize=(5,5))
    plt.imshow(matrix, cmap='gray',aspect='auto')
    plt.ylabel("Secuencia 1")
    plt.xlabel("Secuencia 2")
    plt.savefig(fig_name)

if __name__ == '__main__':
    # Crear el objeto ArgumentParser y definir los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Mi Aplicación MPI')
    parser.add_argument('--file1', type=str, help='Ruta del archivo 1')
    parser.add_argument('--file2', type=str, help='Ruta del archivo 2')
    parser.add_argument('--limite', type=int, help='Umbral')
    #parser.add_argument('--output', type=str, help='Archivo de salida')
    args = parser.parse_args()

    # Obtener los valores de los argumentos
    file1 = args.file1
    file2 = args.file2
    limite = args.limite

    secuencia1 = merge_sequences_from_fasta(file1)
    secuencia2 = merge_sequences_from_fasta(file2)

    begin = time.time()
    dotplot = crear_dotplot(secuencia1[:limite], secuencia2[:limite])
    print("La matriz de resultado tiene tamaño: ", dotplot.shape)
    preview_size = 1000
    dotplot_preview = dotplot[:preview_size, :preview_size].toarray()

    plt.imshow(dotplot_preview, cmap='gray')
    plt.title('Dotplot (Vista previa)')
    plt.xlabel('Secuencia 2')
    plt.ylabel('Secuencia 1')
    plt.savefig('images/Secuencial')
    #draw_dotplot(dotplot_preview, 'images/Secuencial.svg')
    #draw_dotplot(dotplot_preview[:500,:500], 'images/Secuencial_aumentada.svg')

    print(f"\n El código se ejecutó en: {time.time() - begin} segundos")
    print("la matriz resultado tiene un tamaño de " + str(sys.getsizeof(dotplot)) + " bytes")
    #plt.show()
