import time
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO
from mpi4py import MPI


def merge_sequences_from_fasta(file_path):
    sequences = []  # List to store all sequences
    for record in SeqIO.parse(file_path, "fasta"):
        # record.seq gives the sequence
        sequences.append(str(record.seq))
    return "".join(sequences)


def crear_dotplot(args):
    secuencia1, secuencia2, indice = args
    codigos_secuencia1 = np.frombuffer(secuencia1.encode(), dtype=np.uint8)
    codigos_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for i in range(len(secuencia1)):
        matches = codigos_secuencia1[i] == codigos_secuencia2
        dotplot[i, matches] = 1
    return (indice, dotplot)


def procesar_comparacion(secuencia1, secuencia2, chunk_size):
    inicio_parcial = time.time()
    comm = MPI.COMM_WORLD
    num_procesos = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        # Dividir secuencia1 en chunks y enviar a los otros procesos
        subsecuencias1 = dividir_secuencia(secuencia1, chunk_size)
    else:
        subsecuencias1 = None
    # Broadcast de las subsecuencias1 desde el proceso 0 a todos los procesos
    subsecuencias1 = comm.bcast(subsecuencias1, root=0)

    # Cada proceso recibe su subsecuencia1 correspondiente
    subsecuencia1 = subsecuencias1[rank]

    # Cada proceso calcula su parte del dotplot
    resultado_parcial = crear_dotplot((subsecuencia1, secuencia2, rank))

    # Recopilar los resultados parciales en el proceso 0
    resultados = comm.gather(resultado_parcial, root=0)

    if rank == 0:
        dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)

        for i, resultado in tqdm(enumerate(resultados), total=num_procesos):
            indice, resultado_parcial = resultado
            inicio = indice * chunk_size
            fin = min(inicio + chunk_size, len(secuencia1))
            dotplot[inicio:fin] = resultado_parcial
        fin_tiempo_parcial = time.time()
        print("Tiempo parcial: ", fin_tiempo_parcial - inicio_parcial)
        return dotplot
    else:
        return None


def dividir_secuencia(secuencia, chunk_size):
    subsecuencias = []
    start = 0
    while start < len(secuencia):
        end = min(start + chunk_size, len(secuencia))
        subsecuencia = secuencia[start:end]
        subsecuencias.append(subsecuencia)
        start = end
    return subsecuencias


def calcular_peso_matriz(matriz):
    # Obtener el tamaño en bytes de la matriz
    bytes_matriz = sys.getsizeof(matriz)

    # Convertir a megabytes
    megabytes_matriz = bytes_matriz / (1024 ** 2)

    return megabytes_matriz


def draw_dotplot(matrix, fig_name='dotplot.svg'):
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap='gray', aspect='auto')
    plt.ylabel("Secuencia 1")
    plt.xlabel("Secuencia 2")
    plt.savefig(fig_name)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    secuencia1 = merge_sequences_from_fasta('data/E_coli.fna')
    secuencia2 = merge_sequences_from_fasta('data/Salmonella.fna')
    seccion_matriz = 50000
    chunk_size = 1500

    if rank == 0:
        num_procesos = comm.Get_size()
        inicio_tiempo = time.time()
        dotplot = procesar_comparacion(secuencia1[:seccion_matriz], secuencia2[:seccion_matriz], chunk_size)
        preview_size = 30000
        dotplot_preview = dotplot[:preview_size, :preview_size]
        plt.imshow(dotplot_preview, cmap='gray')
        plt.title('Dotplot')
        plt.xlabel('Secuencia 2')
        plt.ylabel('Secuencia 1')
        plt.savefig('images/MPI')
        draw_dotplot(dotplot_preview, 'images/MPI.svg')
        draw_dotplot(dotplot_preview[:500, :500], 'images/MPI_aumentada.svg')
        fin_tiempo = time.time()
        tiempo_ejecucion = fin_tiempo - inicio_tiempo

        print("El código se ejecutó en:", tiempo_ejecucion, "segundos")
        print("El tamaño de la matriz es:", dotplot.shape)
        print("La matriz resultado tiene un tamaño de " + str(calcular_peso_matriz(dotplot)) + " Mb")
    else:
        procesar_comparacion(secuencia1[:seccion_matriz], secuencia2[:seccion_matriz], chunk_size)