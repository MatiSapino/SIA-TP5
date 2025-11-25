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

Puede editar el archivo config.json con los valores deseados para cada Encoder

## Ejecución
Para ejecutar el algoritmo

```
pipenv run python main.py --config-file <config-path> --mode <ae, dae, vae, all>

Ejemplo: para ejecutar solo el Autoencoder básico:

pipenv run python main.py --config-file configs/ae_config.json --mode ae

```


