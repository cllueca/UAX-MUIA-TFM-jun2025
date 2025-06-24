# Código empleado en el TFM en Inteligencia Artificial en la Universidad Alfonso X el Sabio

## Contenido
 - Carpeta `lab`: Código que contiene la preparación de datos, creación, entrenamiento y test de los modelos de visión artificial
 - Carpeta `web`: Código utilizado para desarrollar la aplicación web a modo de demo

## Uso
- ### Mediante python
    * Ir a `web/`
    * Crear un entorno virtual de python `python3 -m venv venv`
    * Activar el entorno virtual `source venv/bin/activate` (linux)
    * Ejecutar la aplicación `python -m src`
    * Una vez levantada, ir a: http://0.0.0.0:11200/
- ### Mediante docker (recomendado)
    * Asegurarse de tener Docker instalado en el sistema
    * Ir a `web/`
    * Construir la imagen `docker build -f docker/Dockerfile -t tfm-demo:v0.0.1 .`
    * Levantar un contenedor
        * Con acceso a GPU (recomendado): `ocker run -d -p 11200:11200 --name tfm-demo-container --gpus all tfm-demo:v0.0.1`
        * Sin acceso a GPU: `ocker run -d -p 11200:11200 --name tfm-demo-container tfm-demo:v0.0.1`
        * Una vez levantada, ir a: http://0.0.0.0:11200/
- ### Utilizar directamente desde Azure (best)
    * Ir a http://4.178.176.157/

## Credenciales
- Contactar conmigo para recibirlas