"""Fórmulas teóricas de las filas M/M/c (con espera) y M/M/c/c (sin espera)."""


def erlang_b(c, a):
    """Calcula la probabilidad de bloqueo B(c, a) con la recurrencia de Erlang B.

    Usa B(0, a) = 1 y B(k, a) = a * B(k-1, a) / (k + a * B(k-1, a)).

    Args:
        c: Número de servidores (camas).
        a: Carga ofrecida, a = lam / mu.

    Returns:
        Probabilidad de que una llegada encuentre todos los servidores ocupados.
    """
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    """Calcula la probabilidad de esperar C(c, a) de Erlang a partir de B(c, a).

    Usa C = B / (1 - rho * (1 - B)), con rho = a / c. Requiere rho < 1.
    """
    rho = a / c
    b = erlang_b(c, a)
    return b / (1 - rho * (1 - b))


def mmc(lam, mu, c):
    """Devuelve las medidas estacionarias de la fila M/M/c.

    Args:
        lam: Tasa de llegadas (por día).
        mu: Tasa de servicio por servidor (por día).
        c: Número de servidores.

    Returns:
        Diccionario con a, rho, C, Lq, Wq, W y L. Los tiempos están en días.
    """
    a = lam / mu
    rho = a / c
    p_esperar = erlang_c(c, a)
    lq = p_esperar * rho / (1 - rho)
    wq = lq / lam
    w = wq + 1 / mu
    return {"a": a, "rho": rho, "C": p_esperar, "Lq": lq, "Wq": wq, "W": w, "L": lam * w}


def mmcc(lam, mu, c):
    """Devuelve las medidas estacionarias de la fila M/M/c/c.

    Args:
        lam: Tasa de llegadas (por día).
        mu: Tasa de servicio por servidor (por día).
        c: Número de servidores.

    Returns:
        Diccionario con a, B, lam_ef (tasa efectiva de ingresos), W (días) y
        L (servidores ocupados en promedio).
    """
    a = lam / mu
    b = erlang_b(c, a)
    lam_ef = lam * (1 - b)
    return {"a": a, "B": b, "lam_ef": lam_ef, "W": 1 / mu, "L": lam_ef / mu}
