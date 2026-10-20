# Hoja de Trabajo 7

Esta actividad modela las camas de cuidados intensivos (UCI) de un hospital con simulación de eventos discretos (grupo 10). Se comparan dos formas de operar: con espera en urgencias (M/M/c) y sin espera, desviando pacientes (M/M/c/c).

- **Task 1 y Task 2** (`task1_2.ipynb`): calcula el desempeño teórico con Erlang B y C, revisa y corrige un simulador generado por IA, lo valida contra la teoría y la ley de Little, y estudia el efecto de la variabilidad de la estancia, de un cierre parcial de camas y de una estancia más larga.

- **Task 3 y Task 4** (`task3_4.ipynb`): determina el número de camas recomendado y su robustez, y acopla una epidemia SIR que aumenta las llegadas (generadas por adelgazamiento) para estimar las camas de contingencia.

Las funciones están en `colas.py` (fórmulas teóricas) y `simulador.py` (simulador corregido). El prompt, el código original de la IA y sus afirmaciones están en `ia/`.

## Cómo Ejecutar

Requisitos: Python 3.14 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Task 1 y Task 2:

```bash
jupyter notebook task1_2.ipynb
```

Task 3 y Task 4:

```bash
jupyter notebook task3_4.ipynb
```
