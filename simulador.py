"""Simulador de eventos discretos de camas UCI: variantes M/M/c (espera) y M/M/c/c (pérdida).

Versión corregida de `ia/simulador_ia.py`. Cambios respecto al original:

- Devuelve también W (tiempo en el sistema), la tasa observada de llegadas y la tasa
  efectiva de ingresos, para verificar la ley de Little solo con datos simulados.
- Registra opcionalmente la longitud de la fila en una rejilla de tiempos, para elegir
  el calentamiento con el promedio móvil entre réplicas.
- Usa `scipy.stats.t` para el cuantil del intervalo de confianza.

Extensiones: distribución de estancia configurable y cierre parcial de camas (Task 2), y
llegadas con tasa variable por adelgazamiento (Task 4).
"""

import heapq
from collections import deque

import numpy as np
from scipy import stats

MEDIDAS = ("L", "Lq", "Wq", "W", "utilizacion", "prob_esperar", "prob_bloqueo",
           "tasa_llegadas", "tasa_ingresos")


def estancia_exponencial(mu):
    """Devuelve un generador de estancias exponenciales de media 1/mu (C_S = 1)."""
    return lambda rng: rng.exponential(1.0 / mu)


def estancia_determinista(mu):
    """Devuelve un generador de estancias fijas iguales a 1/mu (C_S = 0)."""
    return lambda rng: 1.0 / mu


def estancia_lognormal(mu, cs):
    """Devuelve un generador de estancias lognormales de media 1/mu y coeficiente de variación cs.

    Usa s^2 = ln(1 + cs^2) y m = ln(1/mu) - s^2/2 como media y varianza del logaritmo.
    """
    s2 = np.log(1 + cs**2)
    m = np.log(1.0 / mu) - s2 / 2
    return lambda rng: rng.lognormal(m, np.sqrt(s2))


def simular(lam, mu, c, horizonte, calentamiento, semilla, variante, estancia=None, rejilla=None,
            cierre=None, tasa=None, tasa_max=None, historial=False):
    """Simula una réplica y devuelve sus medidas de desempeño.

    Las llegadas ocurren en [0, horizonte). Se observa [calentamiento, horizonte):
    L, Lq y la utilización son promedios ponderados por tiempo en esa ventana; Wq, W y
    las probabilidades usan los pacientes que llegan en ella. Los pacientes que siguen
    dentro al llegar al horizonte terminan su espera y su estancia, y se cuentan completos.

    Args:
        lam: Tasa de llegadas (pacientes/día).
        mu: Tasa de servicio por cama (1/día). Solo se usa si `estancia` es None.
        c: Número de camas.
        horizonte: Fin de las llegadas (días).
        calentamiento: Inicio de la ventana de observación (días).
        semilla: Entero o `np.random.SeedSequence`.
        variante: "espera" (M/M/c) o "perdida" (M/M/c/c).
        estancia: Función `f(rng) -> float` que genera una estancia. Por defecto,
            exponencial de media 1/mu.
        rejilla: Arreglo creciente de tiempos donde se registra la longitud de la fila.
        cierre: Tupla `(t0, t1, k_off)`. En [t0, t1) la capacidad es c - k_off. Los
            pacientes que ya ocupan una cama la conservan hasta su alta; solo se
            bloquean los nuevos ingresos.
        tasa: Función `lam(t)` para llegadas de Poisson no homogéneas, generadas por
            adelgazamiento. Si se pasa, `lam` no se usa para generar llegadas.
        tasa_max: Cota superior de `tasa(t)` en [0, horizonte). Requerida si se pasa `tasa`.
        historial: Si es True, registra el tiempo de cada llegada y si fue desviada.

    Returns:
        Diccionario con las claves de `MEDIDAS` (tiempos en días, tasas por día). Si se
        pasó `rejilla`, incluye "Lq_t" con la fila en cada tiempo de la rejilla. Si se
        pasó `cierre`, incluye "prob_bloqueo_cierre" con la fracción de llegadas en
        [t0, t1) que fueron desviadas. Si `historial` es True, incluye "t_llegadas" y
        "desviado" (arreglos con todas las llegadas, observadas o no).
    """
    if variante not in ("espera", "perdida"):
        raise ValueError("variante debe ser 'espera' o 'perdida'")
    if not 0 <= calentamiento < horizonte:
        raise ValueError("se requiere 0 <= calentamiento < horizonte")
    if estancia is None:
        estancia = estancia_exponencial(mu)

    if not isinstance(semilla, np.random.SeedSequence):
        semilla = np.random.SeedSequence(semilla)
    rng_a, rng_s = (np.random.default_rng(s) for s in semilla.spawn(2))

    SALIDA, REAPERTURA, LLEGADA = 0, 1, 2  # Orden de atención ante un empate de tiempos.
    def siguiente_llegada(t):
        if tasa is None:
            return t + rng_a.exponential(1.0 / lam)
        while True:  # Adelgazamiento: candidatos a tasa_max, aceptados con prob. tasa(t)/tasa_max.
            t += rng_a.exponential(1.0 / tasa_max)
            if t >= horizonte or rng_a.random() * tasa_max <= tasa(t):
                return t

    eventos = [(siguiente_llegada(0.0), LLEGADA, 0.0, False)]
    t_llegadas, desviado = [], []
    if cierre is not None:
        t0, t1, k_off = cierre
        heapq.heappush(eventos, (t1, REAPERTURA, 0.0, False))
    llegadas_cierre = bloqueados_cierre = 0
    cola = deque()
    ocupadas = 0
    anterior = 0.0
    area_n = area_q = area_b = 0.0
    espera_total = sistema_total = 0.0
    llegadas = admitidos = esperan = bloqueados = 0

    lq_t = np.zeros(len(rejilla)) if rejilla is not None else None
    k = 0

    def capacidad(t):
        return c - k_off if cierre is not None and t0 <= t < t1 else c

    def iniciar_servicio(t, llegada, observada):
        heapq.heappush(eventos, (t + estancia(rng_s), SALIDA, llegada, observada))

    def admitir_de_cola(t):
        nonlocal ocupadas, espera_total
        while cola and ocupadas < capacidad(t):
            llegada_cola, obs_cola = cola.popleft()
            ocupadas += 1
            if obs_cola:
                espera_total += t - llegada_cola  # Desde la llegada, no desde el servicio.
            iniciar_servicio(t, llegada_cola, obs_cola)

    while eventos:
        t, tipo, llegada, observada = heapq.heappop(eventos)

        # Estado vigente en [anterior, t), recortado a la ventana de observación.
        dt = max(0.0, min(t, horizonte) - max(anterior, calentamiento))
        area_n += (ocupadas + len(cola)) * dt
        area_q += len(cola) * dt
        area_b += ocupadas * dt
        anterior = t
        if lq_t is not None:
            while k < len(rejilla) and rejilla[k] < t:
                lq_t[k] = len(cola)
                k += 1

        if tipo == LLEGADA:
            observada = t >= calentamiento
            llegadas += observada
            siguiente = siguiente_llegada(t)
            if siguiente < horizonte:
                heapq.heappush(eventos, (siguiente, LLEGADA, 0.0, False))
            en_cierre = cierre is not None and t0 <= t < t1
            llegadas_cierre += en_cierre
            if historial:
                t_llegadas.append(t)
                desviado.append(variante == "perdida" and ocupadas >= capacidad(t))
            if ocupadas < capacidad(t):
                ocupadas += 1
                admitidos += observada
                iniciar_servicio(t, t, observada)
            elif variante == "espera":
                cola.append((t, observada))
                admitidos += observada
                esperan += observada
            else:
                bloqueados += observada
                bloqueados_cierre += en_cierre
        elif tipo == SALIDA:
            if observada:
                sistema_total += t - llegada
            ocupadas -= 1
            admitir_de_cola(t)
        else:
            admitir_de_cola(t)

    duracion = horizonte - calentamiento
    res = dict(zip(MEDIDAS, (
        area_n / duracion,
        area_q / duracion,
        espera_total / admitidos if admitidos else np.nan,
        sistema_total / admitidos if admitidos else np.nan,
        area_b / (c * duracion),
        esperan / llegadas if llegadas else np.nan,
        bloqueados / llegadas if llegadas else np.nan,
        llegadas / duracion,
        admitidos / duracion,
    )))
    if lq_t is not None:
        res["Lq_t"] = lq_t
    if historial:
        res["t_llegadas"], res["desviado"] = np.array(t_llegadas), np.array(desviado)
    if cierre is not None:
        res["prob_bloqueo_cierre"] = bloqueados_cierre / llegadas_cierre if llegadas_cierre else np.nan
    return res


def replicas(lam, mu, c, horizonte, calentamiento, semilla, variante, n_rep, **kwargs):
    """Corre `n_rep` réplicas independientes y resume cada medida.

    Cada réplica recibe una subsemilla de `np.random.SeedSequence(semilla).spawn(n_rep)`.
    La unidad de muestreo del intervalo es la réplica: media ± t(0.975, n-1)·s/√n.

    Args:
        lam, mu, c, horizonte, calentamiento, semilla, variante: Igual que en `simular`.
        n_rep: Número de réplicas (>= 2).
        **kwargs: Argumentos adicionales para `simular` (`estancia`, `rejilla`).

    Returns:
        Tupla `(resumen, datos)`. `resumen[medida]` es un diccionario con "media" e "ic95";
        `datos[medida]` es el arreglo con el valor de cada réplica.
    """
    semillas = np.random.SeedSequence(semilla).spawn(n_rep)
    corridas = [simular(lam, mu, c, horizonte, calentamiento, s, variante, **kwargs) for s in semillas]
    datos = {m: np.array([r[m] for r in corridas]) for m in corridas[0]}
    t_crit = stats.t.ppf(0.975, n_rep - 1)
    resumen = {}
    for m in MEDIDAS:
        x = datos[m]
        media = float(np.mean(x))
        margen = float(t_crit * np.std(x, ddof=1) / np.sqrt(n_rep))
        resumen[m] = {"media": media, "ic95": (media - margen, media + margen)}
    return resumen, datos
