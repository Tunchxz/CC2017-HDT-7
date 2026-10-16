# Afirmaciones

Herramienta utilizada: Codex (modelo GPT-6 Astra).

## Respuesta de la IA

El [código completo en Python](./simulador_ia.py) está implementado y ejecutado. Usa **NumPy como única dependencia externa**, junto con `heapq`, `collections` y `math` de la biblioteca estándar.

Incluye las funciones solicitadas:

```python
simular(lam, mu, c, horizonte, calentamiento, semilla, variante)
replicas(lam, mu, c, horizonte, calentamiento, semilla, variante, n_rep)
```

`simular` devuelve un diccionario con `L`, `Lq`, `Wq`, `utilizacion`, `prob_esperar` y `prob_bloqueo`. `replicas` devuelve la media y el intervalo del 95% de cada medida.

La medición abarca **[200, 2000) días**. `L`, `Lq` y utilización se calculan integrando el estado entre eventos. `Wq` incluye las esperas cero de los admitidos y completa las esperas pendientes después del horizonte, para evitar su truncamiento.

Ejecuté **30 réplicas por variante**, con semillas maestras 2026 y 2027. Los intervalos se calculan entre réplicas mediante:

$
\bar X\pm t_{0.975,29}\frac{s}{\sqrt{30}},
\qquad t_{0.975,29}=2.04523.
$

**Comparación con Erlang C — variante `espera`**

La carga ofrecida es $(a=\lambda/\mu=15.2)$ y $(\rho=a/c=0.894118<1)$, por lo que existe régimen estacionario.

| Medida                  | Media simulada |         IC del 95% |  Teoría |
| ----------------------- | -------------: | -----------------: | ------: |
| $L$, pacientes          |        19.4809 | [18.8812, 20.0806] | 19.9290 |
| $L_q$, pacientes        |         4.3119 |   [3.7839, 4.8400] |  4.7290 |
| $W_q$, días             |         1.1348 |   [0.9994, 1.2702] |  1.2445 |
| Utilización             |         0.8923 |   [0.8871, 0.8975] |  0.8941 |
| Probabilidad de esperar |         0.5508 |   [0.5310, 0.5706] |  0.5600 |
| Probabilidad de bloqueo |              0 |             [0, 0] |       0 |

**Comparación con Erlang B — variante `perdida`**

| Medida                  | Media simulada |         IC del 95% |  Teoría |
| ----------------------- | -------------: | -----------------: | ------: |
| $L$, pacientes          |        13.4179 | [13.3793, 13.4565] | 13.3949 |
| $L_q$, pacientes        |              0 |             [0, 0] |       0 |
| $W_q$, días             |              0 |             [0, 0] |       0 |
| Utilización             |         0.7893 |   [0.7870, 0.7916] |  0.7879 |
| Probabilidad de esperar |              0 |             [0, 0] |       0 |
| Probabilidad de bloqueo |         0.1194 |   [0.1165, 0.1222] |  0.1188 |

En `perdida`, los pacientes desviados cuentan como bloqueados; los admitidos entran directamente y tienen espera cero.

El código calcula las referencias mediante:

$
B=\frac{a^c/c!}{\sum_{k=0}^{c}a^k/k!},
\qquad C=\frac{B}{1-\rho+\rho B}.
$

Para `espera`: $W_q=C/(c\mu-\lambda)$, $L_q=\lambda W_q$, $L=a+L_q$. Para `perdida`: $L=a(1-B)$ y utilización $L/c$. Véase la [referencia de Erlang B y C de Columbia University](https://www.columbia.edu/~ww2040/ErlangBandCFormulas.pdf).

**Sí: las verificaciones respaldan que el simulador es correcto y sus resultados son compatibles con la teoría.** Pasó 64 comparaciones contra un cálculo independiente por paciente, incluyendo sobrecarga y esperas pendientes al cierre. Todos los valores teóricos quedaron dentro de sus respectivos intervalos del 95%. Esto aporta evidencia de corrección; los intervalos son aproximados y marginales, y no constituyen una demostración matemática.
