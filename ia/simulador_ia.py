"""UCI M/M/c y M/M/c/c. Unica dependencia externa: NumPy.

Ejecutar: python simulador_uci.py
El horizonte incluye el calentamiento; se observa [calentamiento, horizonte).
Wq es la espera media de TODOS los admitidos de esa cohorte, incluidos ceros.
La simulacion parte vacia. No se eliminan pacientes al acabar el calentamiento.
"""

import heapq
from collections import deque
from math import lgamma
import numpy as np


MEDIDAS = ("L", "Lq", "Wq", "utilizacion", "prob_esperar", "prob_bloqueo")


def _validar(lam, mu, c, variante):
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lam debe ser finita y >= 0")
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("mu debe ser finita y > 0")
    if isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) or c < 1:
        raise ValueError("c debe ser un entero positivo")
    if variante not in ("espera", "perdida"):
        raise ValueError("variante debe ser 'espera' o 'perdida'")


def _secuencia(semilla):
    # Clonar permite reutilizar un SeedSequence sin alterar su estado de spawn.
    if isinstance(semilla, np.random.SeedSequence):
        return np.random.SeedSequence(semilla.entropy, spawn_key=semilla.spawn_key,
                                      pool_size=semilla.pool_size)
    return np.random.SeedSequence(semilla)


def simular(lam, mu, c, horizonte, calentamiento, semilla, variante):
    """Devuelve seis estimadores; tiempos en dias y utilizacion en [0, 1].

    L, Lq y utilizacion son promedios temporales. Las probabilidades usan
    todas las llegadas observadas. Wq usa los admitidos observados; en perdida
    estos tienen espera cero. Si falta un denominador, se devuelve NaN.
    Tambien permite colas inestables a horizonte finito, sin teoria estacionaria.
    """
    _validar(lam, mu, c, variante)
    if not (np.isfinite(horizonte) and np.isfinite(calentamiento)
            and 0 <= calentamiento < horizonte):
        raise ValueError("Se requiere 0 <= calentamiento < horizonte finito")

    sa, ss = _secuencia(semilla).spawn(2)
    rng_a, rng_s = np.random.default_rng(sa), np.random.default_rng(ss)
    SALIDA, LLEGADA = 0, 1  # Salidas primero ante un empate de tiempos.
    eventos = []
    cola = deque()
    ocupadas = 0
    anterior = 0.0
    area_n = area_q = area_b = espera_total = 0.0
    llegadas = admitidos = esperan = bloqueados = esperas_completas = 0
    total_llegadas = total_admitidos = total_bloqueados = total_salidas = 0

    def programar_llegada(t):
        if lam > 0:
            siguiente = t + rng_a.exponential(1.0 / lam)
            if siguiente < horizonte:
                heapq.heappush(eventos, (siguiente, LLEGADA))

    def programar_salida(t):
        heapq.heappush(eventos, (t + rng_s.exponential(1.0 / mu), SALIDA))

    programar_llegada(0.0)
    while eventos:
        t, tipo = heapq.heappop(eventos)
        # Integrar el estado ANTERIOR al evento y recortar ambos extremos.
        dt = max(0.0, min(t, horizonte) - max(anterior, calentamiento))
        area_n += (ocupadas + len(cola)) * dt
        area_q += len(cola) * dt
        area_b += ocupadas * dt
        anterior = t

        if tipo == LLEGADA:
            total_llegadas += 1
            observada = t >= calentamiento
            llegadas += int(observada)
            programar_llegada(t)
            if ocupadas < c:
                ocupadas += 1
                total_admitidos += 1
                admitidos += int(observada)
                esperas_completas += int(observada)  # Espera cero.
                programar_salida(t)
            elif variante == "espera":
                cola.append((t, observada))
                total_admitidos += 1
                admitidos += int(observada)
                esperan += int(observada)
            else:
                total_bloqueados += 1
                bloqueados += int(observada)
        else:
            total_salidas += 1
            if cola:
                llegada, observada = cola.popleft()
                if observada:
                    espera_total += t - llegada
                    esperas_completas += 1
                programar_salida(t)  # La cama sigue ocupada.
            else:
                ocupadas -= 1

        assert 0 <= ocupadas <= c
        assert not cola or ocupadas == c
        assert variante == "espera" or not cola

    # Se dejan terminar los servicios, sin nuevas llegadas desde horizonte.
    # El estado final es vacio: cualquier tramo final no integrado aporta cero.
    assert total_llegadas == total_admitidos + total_bloqueados
    assert total_admitidos == total_salidas
    assert esperas_completas == admitidos
    assert ocupadas == 0 and not cola
    duracion = horizonte - calentamiento
    return dict(zip(MEDIDAS, (
        area_n / duracion,
        area_q / duracion,
        espera_total / admitidos if admitidos else np.nan,
        area_b / (c * duracion),
        esperan / llegadas if llegadas else np.nan,
        bloqueados / llegadas if llegadas else np.nan,
    )))


def _t975(gl):
    """Cuantil t(0.975), por cuadratura y biseccion; no requiere SciPy."""
    nodos, pesos = np.polynomial.legendre.leggauss(128)
    log_k = lgamma((gl + 1) / 2) - lgamma(gl / 2) - 0.5 * np.log(gl * np.pi)

    def masa_central(x):
        u = (nodos + 1) * x / 2
        densidad = np.exp(log_k - (gl + 1) / 2 * np.log1p(u * u / gl))
        return x / 2 * np.dot(pesos, densidad)

    bajo, alto = 0.0, 1.0
    while masa_central(alto) < 0.475:
        alto *= 2
    for _ in range(60):
        medio = (bajo + alto) / 2
        if masa_central(medio) < 0.475:
            bajo = medio
        else:
            alto = medio
    return (bajo + alto) / 2


def replicas(lam, mu, c, horizonte, calentamiento, semilla, variante, n_rep):
    """Media e IC marginal del 95%: media +/- t * s / sqrt(n_rep).

    Las replicas usan subsemillas independientes y comienzan vacias.
    Los IC t son aproximados porque los estimadores no son exactamente normales.
    No se omiten silenciosamente replicas con medidas indefinidas.
    """
    if (isinstance(n_rep, (bool, np.bool_))
            or not isinstance(n_rep, (int, np.integer)) or n_rep < 2):
        raise ValueError("n_rep debe ser un entero >= 2")
    semillas = _secuencia(semilla).spawn(n_rep)
    resultados = [simular(lam, mu, c, horizonte, calentamiento, s, variante)
                  for s in semillas]
    critico = _t975(n_rep - 1)
    resumen = {}
    for medida in MEDIDAS:
        x = np.array([r[medida] for r in resultados])
        media = float(np.mean(x))
        margen = float(critico * np.std(x, ddof=1) / np.sqrt(n_rep))
        resumen[medida] = {"media": media, "ic95": (media - margen, media + margen)}
    return resumen


def teorico(lam, mu, c, variante):
    """Valores estacionarios mediante Erlang B y Erlang C."""
    _validar(lam, mu, c, variante)
    a = lam / mu
    rho = a / c
    B = 1.0
    for k in range(1, c + 1):
        B = a * B / (k + a * B)
    if variante == "espera":
        if rho >= 1:
            raise ValueError("M/M/c requiere lam < c*mu para tener estado estacionario")
        C = B / (1 - rho + rho * B)
        Wq = C / (c * mu - lam)
        Lq = lam * Wq
        valores = (a + Lq, Lq, Wq, rho, C, 0.0)
    else:
        L = a * (1 - B)
        valores = (L, 0.0, 0.0, L / c, 0.0, B)
    return dict(zip(MEDIDAS, valores))


if __name__ == "__main__":
    print("lam=3.8, mu=0.25, c=17, horizonte=2000, calentamiento=200, n_rep=30")
    print("Semillas: espera=2026, perdida=2027. Wq en dias; utilizacion en fraccion.")
    for variante, semilla in (("espera", 2026), ("perdida", 2027)):
        estimado = replicas(3.8, 0.25, 17, 2000, 200, semilla, variante, 30)
        exacto = teorico(3.8, 0.25, 17, variante)
        print("\n" + variante.upper())
        print(f"{'Medida':<16} {'Media':>10} {'IC95 inferior':>14} {'IC95 superior':>14} {'Teoria':>10} {'Dentro':>8}")
        for medida in MEDIDAS:
            r = estimado[medida]
            inferior, superior = r['ic95']
            dentro = inferior <= exacto[medida] <= superior
            print(f"{medida:<16} {r['media']:10.6f} {inferior:14.6f} {superior:14.6f} {exacto[medida]:10.6f} {str(dentro):>8}")
