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


def procesar_comparacion(secuencia1, secuencia2):
    comm = MPI.COMM_WORLD
    num_procesos = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        # Dividir secuencia1 en partes
        subsecuencias1 = dividir_secuencia(secuencia1, num_procesos)
    else:
        subsecuencias1 = None

    # Enviar subsecuencias1 a todos los procesos
    subsecuencia1 = comm.scatter(subsecuencias1, root=0)

    # Crear array para recibir resultados parciales
    resultado_parcial = np.zeros((len(subsecuencia1), len(secuencia2)), dtype=np.uint8)

    # Convertir secuencia2 una sola vez fuera del bucle
    secuencia2_array = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    # Calcular dotplot para la subsecuencia1
    for i in range(len(subsecuencia1)):
        matches = np.equal(subsecuencia1[i].astype(np.uint8), secuencia2_array)
        resultado_parcial[i, matches] = 1

    # Recopilar los resultados parciales en el proceso 0
    resultados = comm.gather(resultado_parcial, root=0)

    if rank == 0:
        dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)

        # Combinar los resultados parciales en el proceso 0
        for i, resultado in tqdm(enumerate(resultados), total=num_procesos):
            dotplot[i * len(subsecuencia1): (i + 1) * len(subsecuencia1)] = resultado

        return dotplot
    else:
        return None



def dividir_secuencia(secuencia, num_partes):
    longitud_subsecuencia = len(secuencia) // num_partes
    diferencia = len(secuencia) % num_partes  # Diferencia en la longitud que no se divide uniformemente

    subsecuencias = []
    inicio = 0
    for i in range(num_partes):
        fin = inicio + longitud_subsecuencia

        if i < diferencia:
            fin += 1  # Ajustar la longitud de las subsecuencias adicionales

        subsecuencia = secuencia[inicio:fin]
        subsecuencias.append(subsecuencia)

        inicio = fin

    return subsecuencias



def calcular_peso_matriz(matriz):
    # Obtener el tamaño en bytes de la matriz
    bytes_matriz = sys.getsizeof(matriz)

    # Convertir a megabytes
    megabytes_matriz = bytes_matriz / (1024 ** 2)

    return megabytes_matriz


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    secuencia1 = merge_sequences_from_fasta('data/E_coli.fna')
    secuencia2 = merge_sequences_from_fasta('data/Salmonella.fna')

    porcion_matriz = 50000

    if rank == 0:
        num_procesos = comm.Get_size()
        inicio_tiempo = time.time()
        dotplot = procesar_comparacion(secuencia1[0:porcion_matriz], secuencia2[0:porcion_matriz])
        fin_tiempo = time.time()
        tiempo_ejecucion = fin_tiempo - inicio_tiempo

        print("El codigo se ejecuto en:", tiempo_ejecucion, " segundos")
        print("El tamaño de la matriz es: ", dotplot.shape)
        print("La matriz resultado tiene un tamaño de " + str(calcular_peso_matriz(dotplot)) + " Mb")
        preview_size = 1000 
        dotplot_preview = dotplot[:preview_size, :preview_size]
        plt.imshow(dotplot_preview, cmap='gray')
        plt.title('Dotplot')
        plt.xlabel('Secuencia 2')
        plt.ylabel('Secuencia 1')
        plt.savefig('images/MPI')
    else:
        procesar_comparacion(secuencia1[0:porcion_matriz], secuencia2[0:porcion_matriz])
    plt.show()
