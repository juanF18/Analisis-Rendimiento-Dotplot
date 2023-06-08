import multiprocessing
import time
import sys
import numpy as np
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

def crear_dotplot(args):
    secuencia1, secuencia2, indice = args
    codigos_secuencia1 = np.frombuffer(secuencia1.encode(), dtype=np.uint8)
    codigos_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for i in range(len(secuencia1)):
        matches = codigos_secuencia1[i] == codigos_secuencia2
        dotplot[i, matches] = 1
    return (indice, dotplot)


def procesar_comparacion(secuencia1, secuencia2, num_procesos):
    manager = multiprocessing.Manager()
    dotplot = manager.list()

    pool = multiprocessing.Pool(processes=num_procesos)
    subsecuencias1 = dividir_secuencia(secuencia1, num_procesos)
    resultados = [pool.apply_async(crear_dotplot, args=((subseq, secuencia2, i),)) for i, subseq in enumerate(subsecuencias1)]

    for i, resultado in tqdm(enumerate(resultados), total=num_procesos):
        indice, resultado_parcial = resultado.get()
        dotplot.append((indice, resultado_parcial))

    dotplot = sorted(dotplot, key=lambda x: x[0])

    dotplot_final = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for indice, resultado_parcial in dotplot:
        inicio = indice * len(secuencia1) // num_procesos
        fin = (indice + 1) * len(secuencia1) // num_procesos
        dotplot_final[inicio:fin] = resultado_parcial

    pool.close()
    pool.join()

    return dotplot_final

def dividir_secuencia(secuencia, num_partes):
    longitud_subsecuencia = len(secuencia) // num_partes
    subsecuencias = []
    for i in range(num_partes):
        inicio = i * longitud_subsecuencia
        fin = (i + 1) * longitud_subsecuencia
        subsecuencia = secuencia[inicio:fin]
        subsecuencias.append(subsecuencia)
    return subsecuencias

def calcular_peso_matriz(matriz):
    # Obtener el tamaño en bytes de la matriz
    bytes_matriz = sys.getsizeof(matriz)
    
    # Convertir a megabytes
    megabytes_matriz = bytes_matriz / (1024 ** 2)
    
    return megabytes_matriz

def draw_dotplot(matrix, fig_name='dotplot.svg'):
    plt.figure(figsize=(5,5))
    plt.imshow(matrix, cmap='gray',aspect='auto')
    plt.ylabel("Secuencia 1")
    plt.xlabel("Secuencia 2")
    plt.savefig(fig_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mi Aplicación Multiprocessing')
    parser.add_argument('--file1', type=str, help='Ruta del archivo 1')
    parser.add_argument('--file2', type=str, help='Ruta del archivo 2')
    parser.add_argument('--limite', type=int, help='Limite')
    parser.add_argument('--cores', type=int, help='Numero de procesos')

    
    args = parser.parse_args()

    # Obtener los valores de los argumentos
    file1 = args.file1
    file2 = args.file2
    limite = args.limite
    cores = args.cores

    secuencia1 = merge_sequences_from_fasta(file1)
    secuencia2 = merge_sequences_from_fasta(file2)
    num_procesos_range = [1, 2, 4, 8, 10]  # Rango de cantidades de procesos

    tiempos_ejecucion = []  # Lista para almacenar los tiempos de ejecución
    for num_procesos in num_procesos_range:
        inicio_tiempo = time.time()
        dotplot = procesar_comparacion(secuencia1[0:limite], secuencia2[0:limite], num_procesos)
        fin_tiempo = time.time()
        tiempo_ejecucion = fin_tiempo - inicio_tiempo
        tiempos_ejecucion.append(tiempo_ejecucion)
        print("El tiempo con", num_procesos, "procesadores es:", tiempo_ejecucion)
        dotplot = np.zeros([], dtype=np.uint8)

    # Calcular aceleración
    tiempo_secuencial = tiempos_ejecucion[0]  # Tiempo de ejecución secuencial con un solo proceso
    aceleracion = [tiempo_secuencial / tiempo for tiempo in tiempos_ejecucion]

    # Calcular escalabilidad
    escalabilidad = [tiempo_secuencial / (tiempo * num_procesos) for tiempo, num_procesos in zip(tiempos_ejecucion, num_procesos_range)]
    for i, num_procesos in enumerate(num_procesos_range):
        print(f"Cantidad de procesos: {num_procesos}")
        print(f"Aceleración: {aceleracion[i]}")
        print(f"Escalabilidad: {escalabilidad[i]}")
        print("")

    # Crear la figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Graficar los tiempos de ejecución en el subplot izquierdo
    ax1.plot(num_procesos_range, tiempos_ejecucion, marker='o')
    ax1.set_xlabel('Cantidad de procesos')
    ax1.set_ylabel('Tiempo de ejecución (segundos)')
    ax1.set_title('Tiempos de ejecución en función de la cantidad de procesos')

    # Graficar aceleración y escalabilidad en el subplot derecho
    ax2.plot(num_procesos_range, aceleracion, marker='o', label='Aceleración')
    ax2.plot(num_procesos_range, escalabilidad, marker='o', label='Escalabilidad')
    ax2.set_xlabel('Cantidad de procesos')
    ax2.set_ylabel('Valor')
    ax2.set_title('Aceleración y Escalabilidad en función de la cantidad de procesos')
    ax2.legend()

    # Ajustar el espaciado entre subplots
    fig.tight_layout()

    # Guardar la figura
    plt.savefig('images_metricas/Tiempo_Aceleracion_Escalabilidad_Multiprocessing.png')

    # Mostrar la figura
    plt.show()
