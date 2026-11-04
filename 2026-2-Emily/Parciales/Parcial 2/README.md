# Examen Segundo Parcial 02/2026 - Inteligencia Artificial

Este repositorio contiene la solución a los dos ejercicios del examen:

| Ejercicio | Cuadernillo | Tema |
|---|---|---|
| 1 | `01_clasificacion_imagenes.ipynb` | Etiquetado y clasificación de imágenes de 64 x 64 con aprendizaje no supervisado (K-Means) y regresión logística |
| 2 | `02_aprendizaje_refuerzo_pong_hau_ki.ipynb` | Juego Pong Hau K'i aprendido con aprendizaje por refuerzo (función de valor, como el tres en raya de clase) |

Estructura de carpetas:

```
Parcial 2/
├── 01_clasificacion_imagenes.ipynb
├── 02_aprendizaje_refuerzo_pong_hau_ki.ipynb
├── datos/
│   ├── ds1_reducido/   (30 fotos de ds1, carpeta erick_alexander_ochoa_paz, reducidas a 512 px)
│   └── ds2_reducido/   (8 fotos de ds2, reducidas a 512 px)
└── modelos/
    ├── clasificador_parches.pkl     (escalador + K-Means + regresión logística)
    └── agente_pong_hau_ki.pickle    (funciones de valor de los agentes X y O)
```

Para ejecutar: abrir los cuadernillos en Jupyter (Python 3 con `numpy`, `pandas`, `matplotlib`, `pillow`, `scikit-learn` y `tqdm`) y ejecutar todas las celdas. La última celda del ejercicio 2 (`jugar(...)`) es interactiva: pide por teclado qué ficha mover.

---

## Ejercicio 1: Clasificación de imágenes por parches

### Objetivo

El objetivo era construir un programa que ponga una etiqueta al objeto principal de imágenes pequeñas (entre 64 x 64 y 128 x 128 píxeles) y, a partir de esas etiquetas, entrenar un modelo que clasifique imágenes nuevas. Los datasets obligatorios son fotos de quebradas con basura tomadas por los compañeros, así que lo que busco es que el modelo pueda decir, para cada pedazo de una foto, si lo que se ve es **basura**, **vegetación**, **tierra**, **cielo** o **construcción**. Con eso, además, puedo decir qué porcentaje de una foto está cubierto de basura, que es lo útil en la práctica.

### Qué datos usé y por qué

El dataset ds1 está dividido en carpetas por estudiante y no todas tienen el mismo formato (algunas tienen fotos en HEIC o casi vacías). Elegí la carpeta de **erick_alexander_ochoa_paz** porque es la que tiene **más fotos (30)** y todas en JPG, así no tenía que convertir formatos ni mezclar fotos de cámaras muy distintas para entrenar. El dataset ds2 (8 fotos) lo usé al final como **imágenes completamente nuevas**: son de otro lugar y de otra cámara, así que me sirven para ver cómo se comporta el modelo fuera de lo que conoce. De esa manera uso los dos datasets obligatorios.

Las fotos originales son muy grandes (hasta 4096 x 3072 píxeles y casi 18 MB solo la carpeta de ds1, 87 MB ds2). **No cuento con GPU**, así que decidí reducir cada foto para que su lado más largo mida **512 píxeles**. Esto tiene dos ventajas: el procesamiento es rápido en mi computadora y, más importante, a esa escala un parche de 64 x 64 muestra un "objeto" reconocible (un montón de basura, un arbusto, un trozo de cielo) y no un detalle minúsculo como una hoja o una piedra. Después de reducir, ds1 pesa 3 MB y ds2 menos de 1 MB, y por eso pude subir las imágenes reducidas al repositorio.

### Cómo lo hice

**1. Dividir cada foto en imágenes más pequeñas.** Recorté cada foto en una cuadrícula de parches de **64 x 64**. Elegí 64 y no 128 porque con 128 casi todos los parches mezclaban basura, plantas y tierra al mismo tiempo, y no habría un objeto principal claro; además con 64 obtengo **1200 parches** en vez de unos 300, que es importante porque solo tengo 30 fotos. A cada parche le guardé de qué foto viene y en qué fila y columna estaba, para poder separar los datos por foto y para volver a dibujar las etiquetas sobre la foto original.

**2. Describir cada parche con números.** Primero intenté usar directamente los píxeles (como en el ejemplo de MNIST de clase), pero K-Means agrupaba por brillo y por posición ("parches oscuros a la izquierda") y no por lo que había en el parche. En MNIST eso funciona porque los dígitos están centrados; en una foto real la basura puede estar en cualquier lugar del parche. Por eso describí cada parche con **51 características de color y textura**: histogramas de rojo, verde y azul, histograma del tono (pesado por la saturación, para que cuenten los colores vivos de los plásticos y el verde de las plantas), histograma de saturación, media y desviación de cada color y tres medidas de bordes. Estas características no dependen de **dónde** está el objeto dentro del parche, sino de **qué colores y texturas** tiene, que es justo lo que diferencia el cielo (azul y liso), la vegetación (verde), la tierra (marrón y uniforme) y la basura (muchos colores saturados y muchos bordes).

**3. Separar entrenamiento y prueba por foto.** Esta es, para mí, la buena práctica más importante del trabajo. Si mezclara todos los parches y los separara al azar, parches vecinos de una misma foto (misma luz, mismo lugar) quedarían unos en entrenamiento y otros en prueba, y el modelo parecería muy bueno solo porque "reconoce la foto". Eso es fuga de información. Por eso separé **fotos completas**: con `train_test_split` (20% y semilla 42) salieron 6 fotos de prueba, y agregué una más (`img_03`) porque es la única con cielo y construcciones bien visibles; sin ella no podría medir esas clases. Quedaron **23 fotos (912 parches) para entrenar y 7 fotos (288 parches) para probar**, aproximadamente 76/24. Las fotos de prueba no se usan para nada en el entrenamiento: ni para el escalado, ni para K-Means, ni para la regresión logística.

**4. Aprendizaje no supervisado con K-Means.** Con los parches de entrenamiento (sin etiquetas) apliqué K-Means. Revisé el método del codo y el coeficiente de silueta: no hay un codo claro y la silueta es baja (entre 0.13 y 0.20), lo cual es normal en fotos reales, donde los parches no forman grupos perfectamente separados como en los ejemplos sintéticos de clase. Por eso no busqué "el número verdadero de grupos", sino que hice lo mismo que el cuadernillo de clase con los dígitos: usé **k = 50**, muchos más grupos que clases, para que cada grupo sea pequeño y casi todos sus parches sean del mismo tipo.

**5. Etiquetar solo 50 parches y propagar.** De cada grupo tomé el parche **más cercano al centroide** (el representativo) y lo etiqueté yo mirándolo. Así etiqueté a mano solo **50 de 912 parches (5%)**. Luego **propagué** la etiqueta de cada representativo a todo su grupo (aprendizaje semi-supervisado). También probé la propagación parcial (solo al 20% de parches más cercanos a cada centroide), como en clase.

**6. Entrenar el clasificador.** Usé **regresión logística multiclase**, que es el modelo de clasificación visto en clase (one vs all), dentro de un `Pipeline` con `StandardScaler` para que el escalado se aprenda solo con los datos de entrenamiento. Comparé tres versiones: entrenar solo con los 50 representativos, con propagación total y con propagación parcial. Elegí como modelo final la **propagación total** porque usa todos los parches de entrenamiento y es la más estable; la elegí por el método y no por el mejor número de la tabla de prueba, porque elegir mirando la prueba sería ajustar el modelo a esos datos. No usé redes neuronales profundas: no tengo GPU, tengo pocas fotos y una regresión logística con 51 características es suficiente, rápida y fácil de explicar.

### Cómo verifiqué que el modelo funciona

Para medir el modelo **no puedo usar las etiquetas que salieron de K-Means**, porque sería comparar el modelo consigo mismo. Por eso abrí cada una de las 7 fotos de prueba con su cuadrícula de 64 x 64 y **etiqueté a mano cada parche** (287 parches; descarté uno donde aparece una persona). Esa es la "verdad" contra la que comparo. El criterio que usé al etiquetar fue el mismo que en los representativos: es `basura` si los desechos ocupan una parte importante del parche, y `tierra` si es suelo con algún desecho pequeño que no domina.

Las métricas que calculé y por qué:

| Métrica | Resultado | Para qué la uso |
|---|---|---|
| Línea base (decir siempre la clase más común) | 0.35 | Saber si el modelo es mejor que no hacer nada |
| Exactitud en prueba | **0.68** | Qué proporción de parches clasifica bien en fotos nunca vistas |
| F1 macro en prueba | 0.55 | Promedio por clase, castiga si una clase pequeña falla |
| Exactitud en entrenamiento | 0.88 | Compararla con la de validación para ver sobreajuste |
| Validación cruzada por fotos (5 pliegues) | 0.80 ± 0.05 | Ver si el resultado es estable cambiando las fotos de validación |
| Precisión, recall y F1 por clase + matriz de confusión | ver cuadernillo | Saber qué clases funcionan y con qué se confunden |

| Clase | Precisión | Recall | F1 | Parches de prueba |
|---|---|---|---|---|
| basura | 0.65 | 0.73 | 0.68 | 103 |
| vegetación | 0.78 | 0.83 | 0.80 | 100 |
| tierra | 0.60 | 0.51 | 0.55 | 69 |
| cielo | 0.67 | 0.50 | 0.57 | 4 |
| construcción | 0.33 | 0.09 | 0.14 | 11 |

Además de los números, dibujé las etiquetas en color sobre las fotos de prueba (a la izquierda las mías, a la derecha las del modelo) para verificar visualmente dónde acierta y dónde falla, y guardé el modelo con `pickle` y comprobé que al recuperarlo da el mismo resultado.

### Análisis de los resultados

El modelo **es válido**: con una exactitud de 0.68 casi **duplica** la línea base (0.35), y esto se mide con fotos que nunca vio y con etiquetas hechas a mano. Lo más interesante es que con **solo 50 etiquetas humanas** se obtiene prácticamente lo mismo que propagando a los 912 parches (0.69 contra 0.68); eso muestra que el trabajo pesado lo hace el aprendizaje no supervisado, porque K-Means ya organizó los parches en grupos con sentido.

Hay un **sobreajuste moderado**: el modelo acierta 0.88 en entrenamiento y 0.80 en validación cruzada. Es moderado porque la regresión logística es un modelo simple. La diferencia entre la validación cruzada (0.80) y la prueba (0.68) se explica porque la validación se compara con las etiquetas propagadas y la prueba con mis etiquetas manuales: esos 12 puntos son, aproximadamente, el error que se introduce al suponer que todo un grupo tiene la misma etiqueta que su representativo.

Las clases que mejor funcionan son **vegetación y basura**, que tienen colores muy característicos. La confusión más común es **tierra con basura**, y tiene sentido: en estas quebradas la basura está tirada sobre la tierra, así que muchos parches tienen las dos cosas y la frontera es subjetiva incluso para mí al etiquetar. **Cielo y construcción** tienen muy pocos ejemplos (en entrenamiento solo 7 y 16 parches), así que sus métricas son poco confiables; el dataset **no está balanceado** porque en estas fotos casi no aparece cielo ni casas, y no quise inventar ejemplos. La foto de prueba más difícil fue `img_09` (0.35), donde un contenedor de basura verde se confunde con vegetación porque el modelo solo ve color y textura.

Con las fotos de **ds2** el modelo reconoce bien la vegetación verde y detecta la basura, pero **exagera la basura**: ramas secas, sombras y la tierra rojiza de esas fotos se marcan como basura. Esto se llama cambio de dominio: entrené con fotos de una sola persona, con otra cámara y otro tipo de suelo. La forma de corregirlo sería agregar algunas fotos de ds2 al entrenamiento, que es lo que sugiere el enunciado con "agregar más imágenes si se considera necesario".

### Limitaciones y mejoras posibles

Trabajé con pocos datos (30 fotos) a propósito: es lo que tenía la carpeta elegida y alcanza para mostrar el método sin necesitar GPU. Las limitaciones principales son las pocas muestras de cielo y construcción, la subjetividad de la frontera tierra/basura y que mis etiquetas manuales también pueden tener errores. Para mejorar usaría más grupos (por ejemplo k = 100) y más representativos etiquetados, agregaría fotos de otras carpetas de ds1 y de ds2, balancearía las clases y, si tuviera GPU y más fotos, probaría una red convolucional como en el ejemplo de clase con PyTorch.

---

## Ejercicio 2: Aprendizaje por refuerzo con el juego Pong Hau K'i

### Registro del juego

**Pong Hau K'i (juego chino de bloqueo, 5 casillas).** Juego de mesa tradicional chino para dos jugadores en un tablero de solo 5 puntos (4 esquinas y un centro) unidos por líneas. Cada jugador tiene 2 fichas y siempre queda un único punto vacío. En su turno, el jugador mueve una de sus fichas al punto vacío, siempre que esté unido por una línea. Pierde quien en su turno no puede mover (queda bloqueado). Si pasan 20 movimientos sin que nadie quede bloqueado, es empate.

### Por qué elegí este juego

Busqué un juego que fuera **lo más sencillo posible** y que se pudiera programar con el mismo código del tres en raya de clase. Pong Hau K'i es de tablero y por turnos como el tres en raya, pero es todavía más pequeño: solo hay 5 puntos, 4 fichas y como mucho 2 jugadas posibles en cada turno. En total existen solo **30 posiciones distintas** del tablero, así que la tabla de la función de valor es pequeña y la puedo mostrar completa y explicarla fila por fila. Revisé la planilla de juegos registrados y **no se parece a ninguno**: no se colocan fichas para formar líneas (Conecta 4, SOS, Juego de 15), no se capturan piezas (Hexapeón), no se retiran objetos (Nim, Chomp) y no hay dados ni azar (Mini Cacho, Shut the box). Aquí las fichas se deslizan por un tablero en forma de grafo.

### Cómo lo hice

Usé exactamente el método del cuadernillo de clase del tres en raya (`00_intro.ipynb` y `tres.ipynb`): una clase `Board` para el tablero y las reglas, una clase `Game` que hace jugar a dos agentes entre sí (*self-play*) y reparte la recompensa al final (1 si gana, 0.5 si empata, 0 si pierde), y una clase `Agent` que guarda la **función de valor** en un diccionario `estado -> valor`.

El agente juega así: con probabilidad `prob_exp` hace una jugada al azar (**explora**) y si no, prueba cada jugada posible y elige la que lleva al estado de **mayor valor** (**explota**). Al terminar la partida recorre hacia atrás las posiciones que dejó y actualiza cada una con `V(s) = V(s) + alpha * (r - V(s))`, de modo que el resultado final se va propagando a las jugadas que llevaron a él.

La única diferencia con el tres en raya es que en el tres en raya, mirando el tablero, se sabe de quién es el turno (contando X y O), pero en este juego **siempre hay 2 X y 2 O**. Por eso cada agente guarda en su tabla **solo las posiciones que él dejó después de su jugada**; si no, mezclaría "posición donde me toca a mí" con "posición donde le toca al rival" y aprendería valores incorrectos. También agregué el límite de 20 movimientos, porque dos jugadores que no se equivocan pueden mover para siempre.

Elegí los hiperparámetros probando: con exploración de 0.8 (como el agente 1 de clase) el agente X jugaba casi al azar durante el entrenamiento y todavía perdía alrededor del 15% de las partidas. Con **exploración 0.5 y alpha = 0.1** (aprende más lento pero más estable) y **10 000 partidas** de entrenamiento, ningún agente pierde. Entrenar tarda menos de un minuto en mi computadora; no hace falta GPU porque es solo una tabla.

### Cómo verifiqué que el agente aprendió

Comparé contra un **jugador que mueve al azar**, jugando sin exploración, y conté victorias, empates y derrotas antes y después de entrenar. También dibujé la **curva de aprendizaje** (cada 500 partidas de entrenamiento evalúo contra el aleatorio), mostré la **tabla de valores** aprendida y jugué una partida de demostración.

| Jugador (contra un rival aleatorio) | Gana | Empata | Pierde |
|---|---|---|---|
| Aleatorio como X | 0.18 | 0.53 | 0.29 |
| Agente X sin entrenar | 0.00 | 0.41 | 0.59 |
| **Agente X entrenado** | **0.26** | **0.74** | **0.00** |
| Aleatorio como O | 0.28 | 0.55 | 0.18 |
| Agente O sin entrenar | 0.00 | 0.44 | 0.56 |
| **Agente O entrenado** | **0.87** | **0.13** | **0.00** |
| Agente X contra agente O (ambos entrenados) | 0.00 | 1.00 | 0.00 |

### Análisis de los resultados

Después de entrenar, **ninguno de los dos agentes pierde** contra el jugador aleatorio, cuando jugando al azar se perdía entre el 18% y el 29% de las partidas, y el agente sin entrenar (que siempre juega igual) perdía cerca del 58%. La tabla de valores muestra que aprendió el objetivo del juego: por ejemplo, la posición `·OXXO` tiene valor 1.0 para X porque el punto vacío (1) solo se alcanza desde 3 y 4, que son de X, y O queda bloqueado; en cambio `OX·OX` tiene valor 0.03 porque deja el centro libre y O puede encerrar a X, así que el agente aprendió a no dejar nunca esa posición.

El agente O gana mucho más que el X (87% contra 26%). No es un error: en este juego **el que empieza tiene menos oportunidades de encerrar al rival**, y eso ya se ve en la línea base (jugando al azar, X pierde más que O). Cuando los dos agentes entrenados juegan entre sí **siempre empatan**, que es lo que se sabe del Pong Hau K'i: con juego perfecto de ambos lados nadie gana. Todo esto se logró sin técnicas avanzadas: una tabla con 28 de las 30 posiciones posibles y la regla de actualización vista en clase.

---

## Conclusiones del Segundo Parcial

En el ejercicio 1 mostré que con **aprendizaje no supervisado** se puede etiquetar un conjunto de imágenes con muy poco trabajo manual: dividiendo las fotos en parches de 64 x 64, describiéndolos por color y textura y agrupándolos con K-Means, bastó con etiquetar 50 parches para entrenar una regresión logística que acierta el 68% en fotos nunca vistas (casi el doble de la línea base). Lo verifiqué de forma honesta: separando por foto, comparando contra etiquetas hechas a mano y revisando las métricas por clase, que muestran tanto lo que funciona (vegetación y basura) como lo que no (construcción, por falta de ejemplos).

En el ejercicio 2 mostré que un agente puede aprender a jugar **solo con la recompensa al final de la partida**, sin que nadie le diga qué jugada es buena. Usando la misma función de valor del tres en raya de clase, los agentes pasaron de perder más de la mitad de las partidas a no perder nunca contra un jugador aleatorio, y entre ellos llegan al empate, que es el resultado del juego perfecto.

En los dos casos elegí las técnicas más simples vistas en clase (K-Means, regresión logística y función de valor tabular) porque se ajustan a mis recursos (sin GPU y con pocas fotos), son rápidas de entrenar y puedo explicar cada paso en la defensa.
