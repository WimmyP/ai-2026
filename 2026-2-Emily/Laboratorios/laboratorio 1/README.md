# Laboratorio 1 — Mejor línea recta `y = b + w·x`

## Objetivo

Encontrar los mejores valores de **b** (intercepto) y **w** (pendiente) de la ecuación:

```
y = b + w·x
```

a partir de un dataset, **sin** frameworks (scikit-learn, numpy, etc.) y **sin** fórmulas de regresión lineal. Solo estructuras de control **decisivo** (`if`) y **repetitivo** (`for` / `while`).

## Dataset seleccionado

Archivo: `dataset.csv`

Tema: **horas de estudio (x)** → **nota del examen (y)**.

| x (horas) | y (nota) |
|----------:|---------:|
| 1.0 | 52 |
| 1.5 | 55 |
| 2.0 | 58 |
| … | … |
| 8.0 | 93 |

Se eligió este dataset porque la relación es aproximadamente lineal y fácil de interpretar: a más estudio, mayor nota.

## Algoritmo empleado

### Idea general: búsqueda por ensayo y error (rejilla)

1. Se definen rangos posibles para `b` y `w`, y un paso pequeño.
2. Con **bucles anidados** (`while`), se prueba **cada combinación** `(b, w)`.
3. Para cada par, se recorre el dataset y se calcula el **error total**:
   - predicción: `y_pred = b + w * x`
   - error absoluto: `|y_pred - y_real|`
   - se suman todos los errores del dataset.
4. Con **`if`**, si el error actual es menor que el mejor error visto, se actualizan `mejor_b` y `mejor_w`.
5. Al terminar, el par con menor error es la respuesta.

Esto **no** usa la fórmula analítica de mínimos cuadrados; solo prueba y compara.

### Pseudocódigo

```
menor_error ← infinito
para cada b en [b_inicio .. b_fin] con paso b_paso:
    para cada w en [w_inicio .. w_fin] con paso w_paso:
        error_total ← 0
        para cada punto (x, y) del dataset:
            y_pred ← b + w * x
            error_total ← error_total + |y_pred - y|
        si error_total < menor_error:
            menor_error ← error_total
            mejor_b ← b
            mejor_w ← w
```

## Cómo ejecutar

Desde la carpeta `laboratorio 1`:

```bash
cd "laboratorio 1"
python3 buscar_linea.py
```

## Inferencias

Después de encontrar `b` y `w`, el programa estima la nota para **10 nuevos valores de x** (horas de estudio que no están en el dataset), usando:

```
y = mejor_b + mejor_w * x
```

Si la predicción sale fuera de [0, 100], se ajusta a ese rango (una nota no puede ser negativa ni mayor a 100).

## Resultados obtenidos

Al ejecutar el programa se evaluaron **9211** combinaciones `(b, w)`.

| Parámetro | Valor |
|-----------|------:|
| Mejor **b** | 47.00 |
| Mejor **w** | 6.00 |
| Error total | 11.00 |
| Error promedio | 0.73 puntos |

**Ecuación encontrada:** `y = 47.00 + 6.00 · x`

### Interpretación

- **w = 6**: cada hora extra de estudio eleva la nota estimada unos **6 puntos**.
- **b = 47**: con 0 horas de estudio, la nota estimada sería ~47.
- El ajuste es bueno: en el dataset el error medio es menor a 1 punto de nota.

### Inferencias (10 nuevos valores de x)

| # | x (horas) | y predicha |
|--:|----------:|-----------:|
| 1 | 0.5 | 50.00 |
| 2 | 1.2 | 54.20 |
| 3 | 2.8 | 63.80 |
| 4 | 3.3 | 66.80 |
| 5 | 4.7 | 75.20 |
| 6 | 5.2 | 78.20 |
| 7 | 6.8 | 87.80 |
| 8 | 8.5 | 98.00 |
| 9 | 9.0 | 100.00 (tope) |
| 10 | 10.0 | 100.00 (tope) |

Para 9 y 10 horas la predicción cruda supera 100; el programa la limita a 100 porque una nota no puede pasar de ese máximo.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `dataset.csv` | Datos de entrenamiento |
| `buscar_linea.py` | Programa principal |
| `README.md` | Esta explicación |
