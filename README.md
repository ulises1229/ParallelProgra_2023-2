___

# Licenciatura en Tecnología
## Programación Paralela
### Semestre 2023-2


<img src="figs/LogoParallel.png" alt="drawing" style = "text-align: center"/>


___




## Profesor
 Dr. Ulises Olivares Pinto

## Objetivo del curso
Presentar al estudiante el modelo de programación paralela para CPUs y GPUs para la resolución de problemas inherentemente paralelos.


## Prerequisitos
#### Deseables
+ Dominio de los lenguajes de programación C y C++ 
+ Conicimiento básico de estructuras de datos y algoritmos

#### Hardware
Se deberá contar con una computadora con GPU de la marca NVIDIA


#### Software
Se deberá contar con el siguiente software instalado 

  + OpenMP (https://www.openmp.org/)
  + CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
  + Entonrno de desarrollo integrado(IDE)
    - Eclipse
    - Clion
  + [NVIDIA NSIGHT](https://developer.nvidia.com/nsight-visual-studio-edition)
    

#### Cuentas
Se deberán crear cuentas en las siguientes plataformas:
  + Crear una cuenta en GitHub
  
## Contenido del curso
| No.        | Tema           | Conceptos |Código  |  Material complementario|
| :-------------: |:-------------| :-------------|:-----:| :-----|
| 1.              |Introducción a la programación con GPUs          | Introducción a CUDA, modelo de programación paralelo, consulta de dispositivo |   --     |  Artículos<ol><li>[link](https://arxiv.org/abs/1202.4347)</li><li>[link](https://dl.acm.org/doi/abs/10.1145/1365490.1365500)</li><li>[link](https://www.sciencedirect.com/science/article/abs/pii/S0743731508000932)</li></ol>        | 
| 2.              |Modelo de ejecución paralela           |   Hilos, bloques, warps, memoria de dispositivo          | <ul> <li>[vectorAdd.cu](code/vectorAdd)</li> </ul>    | [Capítulos 1 - 3](https://www.iaa.csic.es/~dani/ebooks/MK.Programming.Massively.Parallel.Processors.2nd.Edition.Dec.2012.pdf)| 
| 3.              |Jerarquía de memoria  |  Memoria global, memoria compartida, registros, caches, sincronización de hilos y latencias   |    |  [Capítulo 4](https://www.iaa.csic.es/~dani/ebooks/MK.Programming.Massively.Parallel.Processors.2nd.Edition.Dec.2012.pdf)        |   
| 4.              |Patrones de acceso de memoria: (Convolución)      |   Memoria constante, memoria global, memoria compartida, convoluciones        | <ul> <li>[convolution_global.cu](code/convolution.cu)</li> <li>[convolution_tiled.cu](code/convolution_tiled.cu)</li></ul>       | [Capítulos 5 - 7](https://www.iaa.csic.es/~dani/ebooks/MK.Programming.Massively.Parallel.Processors.2nd.Edition.Dec.2012.pdf)         |    
| 5.              |Patronesd de acceso: Merge Sort     |   Memoria compartida, buffer, algoritmo de ordenamiento     |   <ul><li>[merge.cu](code/merge.cu)</li></ul>     |      [Capítulo 11](https://www.iaa.csic.es/~dani/ebooks/MK.Programming.Massively.Parallel.Processors.2nd.Edition.Dec.2012.pdf)     |    
| 6.              |Memoria unificada     | Inicialización en host y device, prefetching          |  <ul><li>[unifiedMem.cu](code/unifiedMem.cu)</li></ul>        |          |    

##### Última actualización: 10 de diciembre de 2020
