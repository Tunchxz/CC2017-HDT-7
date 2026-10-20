# Bitácora de IA

Herramienta utilizada: Claude Code (modelo Opus 5.5).

## Task 1 y Task 2

### Prompt

```text
Implementa el análisis de camas UCI con teoría de colas y simulación de eventos discretos en dos
módulos, `colas.py` y `simulador.py`, y un Jupyter Notebook `task1_2.ipynb`. Usa NumPy para el
simulador, SciPy solo para el cuantil t del intervalo de confianza y matplotlib solo para graficar.
El código de referencia de la IA está en `ia/simulador_ia.py` y no debe modificarse.

    λ = 3.8 pacientes/día, μ = 0.25 por día, c = 17 camas, k_off = 5, T_b = 7 días
    Variante "espera" (M/M/c) y variante "perdida" (M/M/c/c)

Estructura de `colas.py`:

1. `erlang_b(c, a)` con la recurrencia B(0) = 1, B(k) = a·B(k-1)/(k + a·B(k-1)), y
   `erlang_c(c, a)` a partir de B con C = B/(1 - ρ(1 - B)).
2. `mmc(lam, mu, c)` que devuelva a, ρ, C, Lq, Wq, W y L, y `mmcc(lam, mu, c)` que devuelva
   a, B, la tasa efectiva λ(1 - B), W = 1/μ y L = λ_ef/μ.

Estructura de `simulador.py` (partiendo de la lógica de `ia/simulador_ia.py`):

1. `simular(lam, mu, c, horizonte, calentamiento, semilla, variante, estancia, rejilla, cierre)`
   con lista de eventos en `heapq`, salidas antes que llegadas ante empates y dos generadores
   (llegadas y estancias) creados con `SeedSequence(semilla).spawn(2)`. L, Lq y utilización
   ponderadas por tiempo en [calentamiento, horizonte); Wq y W por paciente desde su llegada,
   completando las esperas y estancias de quienes siguen dentro al llegar al horizonte. Devuelve
   L, Lq, Wq, W, utilización, prob_esperar, prob_bloqueo, tasa_llegadas y tasa_ingresos.
2. `estancia` como función f(rng): `estancia_exponencial(mu)`, `estancia_determinista(mu)` y
   `estancia_lognormal(mu, cs)` con s² = ln(1 + cs²) y m = ln(1/μ) - s²/2.
3. `rejilla` registra la longitud de la fila en esos tiempos (clave "Lq_t").
4. `cierre = (t0, t1, k_off)` reduce la capacidad a c - k_off en [t0, t1) sin expulsar a los
   pacientes en cama, con un evento de reapertura en t1 que admite desde la fila.
5. `replicas(..., n_rep, **kwargs)` con subsemillas de `SeedSequence(semilla).spawn(n_rep)`, que
   devuelva (resumen, datos): media e IC95 con t(0.975, n-1)·s/√n por medida, y el arreglo de
   valores por réplica.

Celdas del notebook:

1. Parámetros, a, ρ y k_off = ⌈c - a⌉ + 3. Medidas teóricas de ambas variantes con tiempos en
   horas, prueba erlang_b(1, a) = a/(1 + a) y ley de Little con λ y con λ_ef.
2. Pruebas sobre `simulador_ia` y `simulador` con un generador falso que devuelve la media en
   `exponential` (con `mock.patch` sobre `np.random.default_rng`), c = 1 y llegadas cada día:
   estancia 0.25 y horizonte 100 (L = 0.2475); estancia 1.5 y horizonte 10.5 con espera
   (Wq = 2.25, Lq = 15/10.5, utilización 9.5/10.5) y sin espera (B = 0.5, utilización 7.5/10.5).
3. M/M/1/1 con λ = 2, μ = 1, horizonte 5000, calentamiento 100, 30 réplicas, semilla 7, contra
   a/(1 + a). Cobertura del IC de B en 100 repeticiones de 30 réplicas (c = 1, λ = μ = 1,
   horizonte 200, calentamiento 20). Número de valores distintos de L en 30 réplicas.
4. Fila en una rejilla diaria de 0 a 3000 días, 100 réplicas, promedio entre réplicas y promedio
   móvil de 30 días con Lq teórico. Guarda la figura en `figuras/`.
5. 30 réplicas por variante con calentamiento 200 y horizonte 5200 (semillas 2026 y 2027). Tabla
   de teoría, media, IC95 y si la teoría cae dentro. Little con L, Lq, W, Wq y las tasas
   simuladas, con su error relativo. Repite `simulador_ia` con horizonte 2000 y compara el
   semiancho relativo del IC de Lq.
6. Allen y Cunneen Wq·(1 + Cs²)/2 para Cs ∈ {0, 1, 2}, T_d = T_b(λ - (c - k_off)μ)/(cμ - λ) y
   erlang_b(c, λ/μ') con 1/μ' = 1.25/μ.
7. Variante con espera con las tres estancias (semillas 300 + Cs), Wq en horas con IC95 y error
   relativo contra Allen y Cunneen.
8. Cierre en [300, 307) con 1000 réplicas, horizonte 600 y rejilla cada 0.1 días en [250, 550),
   con estancia exponencial y determinista. Recuperación: primer tiempo tras la reapertura en que
   la fila media es menor que 1.1·Lq (teórico para la exponencial, simulado para la determinista).
   IC95 por bootstrap con 200 remuestreos de réplicas. Gráfica con el cierre sombreado. Guarda la
   figura en `figuras/`.
9. Variante sin espera con μ = 0.2 y 30 réplicas, contra la predicción de Erlang B.
```

### ¿Por qué funcionó?

Funcionó porque el prompt fija los parámetros del grupo, el calentamiento, el horizonte, el número de réplicas y la semilla de cada experimento. No queda margen para que una celda use otra ventana de observación u otra cantidad de réplicas, y los valores teóricos de comparación se calculan con las mismas funciones de `colas.py` en todo el notebook.

También define cómo se miden las cantidades antes de escribir el simulador. Al pedir promedios ponderados por tiempo, esperas desde la llegada y pacientes completos al final del horizonte, los siete puntos de revisión se pueden comprobar en el código y en las pruebas. La decisión clave es que `simular` devuelva $W$ y las tasas observadas. Con eso, la ley de Little se verifica solo con datos simulados, y los argumentos `estancia`, `rejilla` y `cierre` permiten reutilizar el mismo simulador en el Task 2 sin copiarlo.

Finalmente, los casos que suelen fallar se resuelven dentro del propio prompt. El generador falso convierte el simulador en un caso determinista con respuesta conocida sin cambiar su código, y usar dos generadores derivados de la semilla mantiene las mismas llegadas al cambiar la distribución de la estancia. Y las 1000 réplicas con bootstrap evitan que el ruido de la curva promedio, que se acerca muy despacio al umbral de $1.1\,L_q$, decida por sí solo el tiempo de recuperación.

## Task 3 y Task 4

### Prompt

```text
Implementa el dimensionamiento de camas y el acople con una epidemia en un Jupyter Notebook
`task3_4.ipynb`, reutilizando `colas.py` y `simulador.py`. Usa NumPy para el modelo SIR, SciPy
solo para el cuantil t y matplotlib solo para graficar.

    λ = 3.8, μ = 0.25, k_off = 5, T_b = 7, calentamiento 200, horizonte 5200, 30 réplicas
    Metas: B ≤ 0.02 (sin espera) y Wq ≤ 6 h (con espera)
    SIR: N = 500000, γ = 0.2, β = 0.32, I(0) = 100, R(0) = 0, p_h = 0.0002

Extiende `simulador.simular` con:

1. `tasa` y `tasa_max` para llegadas no homogéneas por adelgazamiento: candidatos exponenciales a
   tasa_max, aceptados con probabilidad tasa(t)/tasa_max. Sin `tasa`, las llegadas no cambian.
2. `historial=True` que devuelva el tiempo de cada llegada y si fue desviada.
3. Con `cierre`, la clave "prob_bloqueo_cierre" con la fracción desviada de las llegadas en
   [t0, t1).

Celdas del notebook:

1. `camas_bloqueo(lam, mu)` y `camas_espera(lam, mu, f_wq)` que busquen el menor c que cumple
   cada meta, donde f_wq multiplica Wq. Tabla de B y Wq para c = 17 a 25.
2. `verificar(variante, c, semilla, estancia)` que imprima la medida de la meta con su IC95 y si
   cumple (IC bajo la meta), no cumple (IC sobre la meta) o es dudoso. Aplícala al mínimo de cada
   meta y a una cama menos.
3. Mínimos para la base, estancia +25% (μ/1.25), llegadas +15% (1.15·λ) y Cs = 2 (f_wq = 2.5).
   Verifica con estancia lognormal Cs = 2 el mínimo con espera, una cama menos y el mínimo sin
   espera.
4. Con C_REC = 22, variante sin espera y cierre en [300, 307) con 1000 réplicas y horizonte 308:
   B durante el cierre con IC95, Erlang B con C_REC - k_off y C_REC, desviados esperados y camas
   temporales que faltan para cada meta.
5. SIR integrado con RK4 propio (h = 0.01, 200 días) y λ_epi = p_h·β·S·I/N. Día y valor del pico
   y pacientes adicionales por `np.trapezoid`. Gráfica de λ_epi, λ + λ_epi y la tasa base.
   Guarda la figura en `figuras/`.
6. Epidemia desde el día 100 (antes, tasa base) con `tasa(t)` por `np.interp` y tasa_max igual al
   máximo de λ(t). 100 réplicas sin espera con C_REC = 22 y `historial=True`. Ola: días con
   λ_epi ≥ 10% del pico. Desviados y fracción desviada en la ola por réplica, con IC95.
7. Fracción diaria desviada sumando todas las réplicas con `np.add.at`, con la ola sombreada y la
   referencia de Erlang B. Guarda la figura en `figuras/`.
8. Aumenta c de uno en uno desde C_REC (100 réplicas, semilla 3000 + c) hasta que la fracción
   media desviada en la ola sea ≤ 5%. Imprime cada c con su IC95.
```

### ¿Por qué funcionó?

Funcionó porque el prompt entrega las dos metas, los tres escenarios y los parámetros del SIR como dato duro. No queda margen para escoger otra forma de medir el cumplimiento, y la regla de `verificar` (cumple solo si todo el IC queda bajo la meta) se define antes de ver cualquier resultado, con lo que los casos al límite aparecen como dudosos en lugar de ocultarse.

También fija las funciones de búsqueda para que cada escenario sea una llamada de una línea. `camas_bloqueo` y `camas_espera` recorren $c$ con las fórmulas de `colas.py`, y el factor `f_wq` permite usar la aproximación de Allen y Cunneen sin escribir otra función. La decisión clave es extender `simular` con `tasa`, `tasa_max` e `historial` en lugar de crear un simulador nuevo, con eso la ola usa exactamente el mismo motor validado en el Task 1 y la búsqueda de camas de contingencia es un ciclo corto.

Finalmente, el costo computacional y el ruido se controlan dentro del propio prompt. Integrar el SIR una sola vez e interpolar $\lambda_{epi}$ evita resolver la epidemia en cada llegada, y sumar llegadas y desvíos de todas las réplicas por día da una curva de bloqueo estable sin guardar trayectorias. Y la ventana de la ola definida sobre el pico de $\lambda_{epi}$ hace que el total de desviados se compare siempre sobre los mismos días al cambiar el número de camas.
