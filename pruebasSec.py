import sys
import time
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from Bio import SeqIO

def merge_sequences_from_fasta(file_path):
    sequences = []  # List to store all sequences
    for record in SeqIO.parse(file_path, "fasta"):
        # record.seq gives the sequence
        sequences.append(str(record.seq))
    return "".join(sequences)

def crear_dotplot(secuencia1, secuencia2):
    codigos_secuencia1 = np.frombuffer(secuencia1.encode(), dtype=np.uint8)
    codigos_secuencia2 = np.frombuffer(secuencia2.encode(), dtype=np.uint8)

    dotplot = lil_matrix((len(secuencia1), len(secuencia2)), dtype=np.uint8)
    for i in tqdm(range(len(secuencia1))):
        matches = codigos_secuencia1[i] == codigos_secuencia2
        dotplot[i, matches] = 1
    return dotplot

secuencia1 = merge_sequences_from_fasta("./data/E_coli.fna")
secuencia2 = merge_sequences_from_fasta("./data/Salmonella.fna")

begin = time.time()
dotplot = crear_dotplot(secuencia1[:30000], secuencia2[:30000])
print("La matriz de resultado tiene tamaño: ", dotplot.shape)
preview_size = 1000
dotplot_preview = dotplot[:preview_size, :preview_size].toarray()

plt.imshow(dotplot_preview, cmap='gray')
plt.title('Dotplot (Vista previa)')
plt.xlabel('Secuencia 2')
plt.ylabel('Secuencia 1')
plt.savefig('images/Secuencial')

print(f"\n El código se ejecutó en: {time.time() - begin} segundos")
print("la matriz resultado tiene un tamaño de " + str(sys.getsizeof(dotplot)) + " bytes")
plt.show()
