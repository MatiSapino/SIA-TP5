# TP5 SIA - Deep Learning

## Introducción

Este trabajo práctico implementa distintos modelos de Aprendizaje No Supervisado, enfocados en la compresión, reconstrucción y generación de datos utilizando:

Autoencoder básico (AE)
Denoising Autoencoder (DAE)
Variational Autoencoder (VAE)

El objetivo es explorar la representación latente de caracteres de una fuente bitmap, la reconstrucción de caracteres ruidosos y la generación de dígitos sintéticos a partir del espacio latente del VAE.

[Enunciado](Enunciado.pdf)

[Presentación](Presentacion.pdf)

### Requisitos

- Python3
- pip3
- [pipenv](https://pypi.org/project/pipenv/)

### Instalación

Parado en la carpeta del tp4 ejecutar

```sh
  pipenv install -r requirements.txt
```

Para instalar las dependencias necesarias en el ambiente virtual

## Ejecución
Para ejecutar el algoritmo
```
pipenv run python main.py <PARAMETRO>

| Parámetro           | Descripción                                                     |
| ------------------- | --------------------------------------------------------------- |
| `--run-ae`          | Ejecuta solo la parte del Autoencoder básico.                   |
| `--run-dae`         | Ejecuta solo la parte del Denoising Autoencoder.                |
| `--run-vae`         | Ejecuta solo la parte del VAE.                                  |
| `--all` *(default)* | Ejecuta las tres partes.                                        |
| `--mnist-samples N` | Especifica cuántas imágenes de MNIST usar para entrenar el VAE. |

```
