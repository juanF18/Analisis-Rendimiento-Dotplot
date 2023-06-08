import time
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO
from mpi4py import MPI
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mi Aplicación MPI')
    parser.add_argument('--file1', type=str, help='Ruta del archivo 1')
    parser.add_argument('--file2', type=str, help='Ruta del archivo 2')
    parser.add_argument('--limite', type=int, help='Numero de procesos')
    
    args = parser.parse_args()

    # Obtener los valores de los argumentos
    file1 = args.file1
    file2 = args.file2
    limite = args.limite

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    secuencia1 = merge_sequences_from_fasta(file1)
    secuencia2 = merge_sequences_from_fasta(file2)
    seccion_matriz = limite
    chunk_size = 1000

    # Listas para almacenar los tiempos de ejecución y los números de procesos
    tiempos = []
    procesos = []

    for num_procesos in range(1, comm_size + 1):
        comm.Barrier()  # Sincronizar todos los procesos

        if rank == 0:
            inicio_tiempo = time.time()

        if num_procesos == 1:
            dotplot = procesar_comparacion(secuencia1[0:seccion_matriz], secuencia2[0:seccion_matriz], chunk_size)
        else:
            dotplot = None

        comm.Barrier()  # Sincronizar todos los procesos

        if rank == 0:
            fin_tiempo = time.time()
            tiempo_ejecucion = fin_tiempo - inicio_tiempo

            tiempos.append(tiempo_ejecucion)
            procesos.append(num_procesos)

            # Imprimir el tiempo de ejecución para el número actual de procesos
            print(f"Tiempo de ejecución con {num_procesos} proceso(s): {tiempo_ejecucion} segundos")

    if rank == 0:
        # Graficar los tiempos de ejecución, aceleración y escalabilidad
        plt.figure(figsize=(15, 5))

        # Gráfica de tiempos
        plt.subplot(1, 3, 1)
        plt.plot(procesos, tiempos)
        plt.xlabel("Número de procesos")
        plt.ylabel("Tiempo de ejecución (segundos)")
        plt.title("Tiempos de ejecución con diferentes números de procesos")

        # Gráfica de aceleración
        plt.subplot(1, 3, 2)
        aceleracion = [tiempos[0] / t for t in tiempos]
        plt.plot(procesos, aceleracion)
        plt.xlabel("Número de procesos")
        plt.ylabel("Aceleración")
        plt.title("Aceleración con diferentes números de procesos")

        # Gráfica de escalabilidad
        plt.subplot(1, 3, 3)
        escalabilidad = [tiempos[0] / (t * p) for t, p in zip(tiempos, procesos)]
        plt.plot(procesos, escalabilidad)
        plt.xlabel("Número de procesos")
        plt.ylabel("Escalabilidad")
        plt.title("Escalabilidad con diferentes números de procesos")

        plt.tight_layout()
        plt.show()
