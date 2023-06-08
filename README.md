# Analisis de Rendimiento de Dotplot Secuencial vs. Paralelizacion

## Descripcion general

---

El objetivo de este proyecto es implementar y analizar el rendimiento de tres formas de realizar un dotplot, una técnica comúnmente utilizada en bioinformática para comparar secuencias de ADN o proteínas.

### Nota:

Recordar que se tiene que crear un entorno virutal con

    python3 -m venv .venv -> Linux
    python -m venv .venv -> Windows

Tenemos que instalar las librerias:

    pip install numpy
    pip install matplotlib
    pip install biopython
    pip install tqdm
    pip install mpi4py

Antes de empezar a ejecutar los comandos tenemos que tener en cuenta como son los comandos por que tenemos parametros para poder utilizar el codigo con facilida

- --file1 (ruta del primer archivo)
- --file2 (ruta del segundo archivo)
- --limite (el limite de carcateres en ambas matrices)

  - Esto se hacer para que la maquina no se bloquee por calcular una matriz muy grande

- --cores (numero de procesos que se van a utilizar `Este solo se utiliza en multiprocessing`)

## Ejemplos de ejecucion

---

Para ejecutar el codgio en secuencial utilizamos el comando

    python .\pruebasSec.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --limite=50000

Para ejecutar el codgio en Multiprocessing utilizamos el comando

    python .\pruebasMulti.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --limite=50000 --cores=10

Para ejecutar el codgio con MPI utilizamos el comando

    mpiexec -n 10  python .\pruebasMPI.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --limite=70000

Para ejecutar las metricas de MPI utilizamos el comando

     mpiexec -n 10  python .\metricas\metricas_MPI.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --limite=50000

Para ejecutar el codigo de metricas con Multiprocessing utilizamos el comando

    python .\metricas\metricas_multiprocessing.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --limite=50000 --cores=10

## Resultados

---

Estos los podemos visualizar en la `carpeta de imagenes`, `imagenes_metricas` y `filtro`

## Tambien recuerde descargar los .fna desde el drive

---

### [Link](https://drive.google.com/drive/folders/1rgurfK3JdKO1vRtTsGP0dm93kxSKyN3s?usp=sharing)
