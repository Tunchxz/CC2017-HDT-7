# Prompt (Task 1.2)

```text
Escribe en Python (solo numpy) un simulador de eventos discretos para una unidad de
cuidados intensivos con c camas. Los pacientes llegan según un proceso de Poisson de tasa
lam (pacientes/día) y cada uno ocupa una cama un tiempo exponencial de media 1/mu días.
Debe simular dos variantes:
- "espera" (M/M/c): si no hay cama libre, el paciente espera en fila FIFO hasta que se libere una.
- "perdida" (M/M/c/c): si no hay cama libre, el paciente es desviado y se pierde.

Requisitos:
1. Usa una lista de eventos (heapq) con eventos de llegada y salida.
2. Función simular(lam, mu, c, horizonte, calentamiento, semilla, variante) que devuelva
   L, Lq, Wq, utilización, probabilidad de esperar y probabilidad de bloqueo.
3. Función replicas(..., n_rep) que corra varias réplicas y devuelva media e intervalo de
   confianza del 95% de cada medida.
4. Ejecútalo con lam = 3.8, mu = 0.25, c = 17, horizonte = 2000 días, calentamiento = 200
   días, 30 réplicas, y compara contra las fórmulas teóricas de Erlang C y Erlang B.
Al final, dime si el simulador es correcto y si sus resultados coinciden con la teoría.
```
