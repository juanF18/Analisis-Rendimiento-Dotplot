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
        # Dividir secuencia1 en partes y calcular el tamaño de cada parte
        longitud_secuencia1 = len(secuencia1)
        division = longitud_secuencia1 // num_procesos
        tamanos = [division] * (num_procesos - 1)
        tamanos.append(longitud_secuencia1 - division * (num_procesos - 1))

        # Calcular los desplazamientos para el Scatterv
        desplazamientos = [0]
        for t in tamanos[:-1]:
            desplazamientos.append(desplazamientos[-1] + t)

        # Calcular los tamaños totales y desplazamientos para el Gatherv
        total_tamanos = comm.gather(len(secuencia1), root=0)
        total_desplazamientos = comm.gather(desplazamientos[-1] + tamanos[-1], root=0)
    else:
        tamanos = None
        desplazamientos = None
        total_tamanos = None
        total_desplazamientos = None

    # Broadcast de los tamaños y desplazamientos a todos los procesos
    tamanos = comm.bcast(tamanos, root=0)
    desplazamientos = comm.bcast(desplazamientos, root=0)
    total_tamanos = comm.bcast(total_tamanos, root=0)
    total_desplazamientos = comm.bcast(total_desplazamientos, root=0)

    # Cada proceso recibe su subsecuencia1 correspondiente
    subsecuencia1 = np.empty(tamanos[rank], dtype='U1')
    comm.Scatterv([secuencia1, tamanos, desplazamientos, MPI.CHAR], subsecuencia1, root=0)

    # Cada proceso calcula su parte del dotplot
    resultado_parcial = crear_dotplot((subsecuencia1, secuencia2, rank))

    # Recopilar los resultados parciales en el proceso 0
    resultados = comm.gather(resultado_parcial, root=0)

    if rank == 0:
        dotplot = np.zeros((total_tamanos[0], len(secuencia2)), dtype=np.uint8)
        indices = [r[0] for r in resultados]
        resultados_parciales = [r[1] for r in resultados]
        print("Iniciando procesamiento de comparación...")
        for i, resultado in tqdm(enumerate(zip(indices, resultados_parciales)), total=num_procesos):
            indice, resultado_parcial = resultado
            dotplot[desplazamientos[indice]:desplazamientos[indice] + len(resultado_parcial)] = resultado_parcial
        print("Fin procesamiento de comparación...")
        return dotplot
    else:
        return None

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
    seccion_matriz = 45000

    if rank == 0:
        num_procesos = comm.Get_size()
        inicio_tiempo = time.time()
        dotplot = procesar_comparacion(secuencia1[0:seccion_matriz], secuencia2[0:seccion_matriz])
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
        procesar_comparacion(secuencia1[0:seccion_matriz], secuencia2[0:seccion_matriz])
    plt.show()
