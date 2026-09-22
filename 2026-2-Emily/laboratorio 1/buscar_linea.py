"""
Laboratorio 1 - Búsqueda de los mejores valores b y w
para la ecuación de la línea recta: y = b + w*x

No usa frameworks ni fórmulas de regresión.
Solo estructuras de control decisivo (if) y repetitivo (for/while).
"""

# ------------------------------------------------------------
# 1. Cargar el dataset (horas de estudio -> nota del examen)
# ------------------------------------------------------------
x_valores = []
y_valores = []

archivo = open("dataset.csv", "r")
primera_linea = True

for linea in archivo:
    linea = linea.strip()
    if primera_linea:
        primera_linea = False
        continue
    if linea == "":
        continue

    partes = linea.split(",")
    x = float(partes[0])
    y = float(partes[1])
    x_valores.append(x)
    y_valores.append(y)

archivo.close()

n = len(x_valores)
print("=" * 60)
print("DATASET: horas de estudio vs nota del examen")
print("=" * 60)
print(f"Cantidad de puntos: {n}")
print()
print(f"{'x (horas)':>12}  {'y (nota)':>12}")
print("-" * 28)
i = 0
while i < n:
    print(f"{x_valores[i]:12.1f}  {y_valores[i]:12.1f}")
    i = i + 1
print()

# ------------------------------------------------------------
# 2. Algoritmo: búsqueda por ensayo y error (rejilla)
#
#    Se prueban muchos pares (b, w). Para cada par se calcula
#    el error total (suma de errores absolutos) al predecir
#    cada y del dataset. Se conserva el par con menor error.
# ------------------------------------------------------------

# Rangos de búsqueda (ajustables)
b_inicio = 30.0
b_fin = 60.0
b_paso = 0.5

w_inicio = 0.0
w_fin = 15.0
w_paso = 0.1

mejor_b = None
mejor_w = None
menor_error = None
intentos = 0

print("=" * 60)
print("BUSCANDO mejores b y w (y = b + w*x)")
print("=" * 60)
print(f"Rango de b: de {b_inicio} a {b_fin} (paso {b_paso})")
print(f"Rango de w: de {w_inicio} a {w_fin} (paso {w_paso})")
print()

b = b_inicio
while b <= b_fin:
    w = w_inicio
    while w <= w_fin:
        # Calcular el error total de este par (b, w)
        error_total = 0.0
        i = 0
        while i < n:
            y_predicho = b + w * x_valores[i]
            diferencia = y_predicho - y_valores[i]
            if diferencia < 0:
                diferencia = -diferencia  # valor absoluto
            error_total = error_total + diferencia
            i = i + 1

        # Decisión: ¿este par es mejor que el mejor conocido?
        if menor_error is None:
            menor_error = error_total
            mejor_b = b
            mejor_w = w
        else:
            if error_total < menor_error:
                menor_error = error_total
                mejor_b = b
                mejor_w = w

        intentos = intentos + 1
        w = w + w_paso
    b = b + b_paso

print(f"Combinaciones evaluadas: {intentos}")
print()
print("MEJORES VALORES ENCONTRADOS:")
print(f"  b (intercepto) = {mejor_b:.2f}")
print(f"  w (pendiente)  = {mejor_w:.2f}")
print(f"  Error total    = {menor_error:.2f}")
print(f"  Error promedio = {menor_error / n:.2f} puntos de nota")
print()
print(f"Ecuación de la línea:  y = {mejor_b:.2f} + {mejor_w:.2f} * x")
print()

# ------------------------------------------------------------
# 3. Comparar predicciones vs valores reales del dataset
# ------------------------------------------------------------
print("=" * 60)
print("AJUSTE SOBRE EL DATASET (predicción vs real)")
print("=" * 60)
print(f"{'x':>8}  {'y real':>8}  {'y pred':>8}  {'|error|':>8}")
print("-" * 40)

i = 0
while i < n:
    y_pred = mejor_b + mejor_w * x_valores[i]
    err = y_pred - y_valores[i]
    if err < 0:
        err = -err
    print(f"{x_valores[i]:8.1f}  {y_valores[i]:8.1f}  {y_pred:8.2f}  {err:8.2f}")
    i = i + 1
print()

# ------------------------------------------------------------
# 4. Inferencias para 10 nuevos valores de x
# ------------------------------------------------------------
print("=" * 60)
print("INFERENCIAS PARA 10 NUEVOS VALORES DE x")
print("=" * 60)

# Nuevos valores de horas de estudio (no estaban en el dataset)
nuevos_x = [0.5, 1.2, 2.8, 3.3, 4.7, 5.2, 6.8, 8.5, 9.0, 10.0]

print(f"{'#':>4}  {'x (horas)':>12}  {'y predicha (nota)':>20}")
print("-" * 42)

k = 0
while k < 10:
    x_nuevo = nuevos_x[k]
    y_nuevo = mejor_b + mejor_w * x_nuevo

    # Limitar la nota a un rango razonable [0, 100]
    if y_nuevo < 0:
        y_mostrar = 0.0
        nota_extra = " (ajustada a 0)"
    else:
        if y_nuevo > 100:
            y_mostrar = 100.0
            nota_extra = " (ajustada a 100)"
        else:
            y_mostrar = y_nuevo
            nota_extra = ""

    print(f"{k + 1:4d}  {x_nuevo:12.1f}  {y_mostrar:20.2f}{nota_extra}")
    k = k + 1

print()
print("=" * 60)
print("EXPLICACIÓN BREVE DE LOS RESULTADOS")
print("=" * 60)
print(
    "Se eligió un dataset de horas de estudio (x) y nota de examen (y).\n"
    "El algoritmo probó muchas combinaciones de b y w con bucles anidados\n"
    "y se quedó con el par que produce el menor error absoluto total.\n"
    f"La pendiente w ≈ {mejor_w:.2f} indica que, en promedio, cada hora\n"
    f"extra de estudio eleva la nota unos {mejor_w:.2f} puntos.\n"
    f"El intercepto b ≈ {mejor_b:.2f} es la nota estimada si x = 0\n"
    "(cero horas de estudio).\n"
    "Las 10 inferencias usan la misma ecuación y = b + w*x para\n"
    "estimar notas de horas de estudio que no estaban en el dataset."
)
print("=" * 60)
