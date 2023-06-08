import multiprocessing
import time
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO

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
    pool = multiprocessing.Pool(processes=num_procesos)
    subsecuencias1 = dividir_secuencia(secuencia1, num_procesos)
    resultados = [pool.apply_async(crear_dotplot, args=((subseq, secuencia2, i),)) for i, subseq in enumerate(subsecuencias1)]
    dotplot = np.zeros((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for i, resultado in tqdm(enumerate(resultados), total=num_procesos):
        indice, resultado_parcial = resultado.get()
        dotplot[indice * len(secuencia1) // num_procesos: (indice + 1) * len(secuencia1) // num_procesos] = resultado_parcial
    pool.close()
    pool.join()
    return dotplot

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
    secuencia1 = merge_sequences_from_fasta('data/E_coli.fna')
    secuencia2 = merge_sequences_from_fasta('data/Salmonella.fna')
    num_procesos = 10

    inicio_tiempo = time.time()
    dotplot = procesar_comparacion(secuencia1[0:50000], secuencia2[0:50000], num_procesos)
    preview_size = 30000 
    dotplot_preview = dotplot[:preview_size, :preview_size]
    plt.imshow(dotplot_preview, cmap='gray', aspect='auto')
    plt.title('Dotplot (Vista previa)')
    plt.xlabel('Secuencia 2')
    plt.ylabel('Secuencia 1')
    plt.savefig('images/Multiprocessing')
    draw_dotplot(dotplot_preview, 'images/Multiprocessing.svg')
    draw_dotplot(dotplot_preview[:500,:500], 'images/Multiprocessing_aumentada.svg')
    fin_tiempo = time.time()
    tiempo_ejecucion = fin_tiempo - inicio_tiempo

    print("El codigo se ejecuto en:", tiempo_ejecucion, " segundos")
    print("El tamaño de la matriz es: ",dotplot.shape)
    print("La matriz resultado tiene un tamaño de " + str(calcular_peso_matriz(dotplot)) + " Mb")
    #plt.show()