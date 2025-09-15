"""
Apendice A – Simulacao de hotspots em placas eletronicas
Autor: Ivan de Oliveira Bessa RU: 3613650
Curso: Engenharia da Computacao
Instituicao: UNINTER
Ano: 2025

Descricao:
-----------
Este codigo implementa uma simulacao 2D da conducao de calor em regime estacionario
utilizando o metodo das diferenças finitas.
O objetivo eh avaliar o efeito do aumento da condutividade termica efetiva (k) de
materiais polimericos quando dopados com Flash Grafeno (FG).
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. Parametros do dominio
# -----------------------------
Lx, Ly = 0.10, 0.10     # dimensoes da placa (10 cm x 10 cm)
nx, ny = 120, 120       # numero de pontos da malha
dx, dy = Lx/(nx-1), Ly/(ny-1)
T_amb = 25.0            # temperatura ambiente (°C, condicao de contorno)

# -----------------------------
# 2. Fonte de calor (chip)
# -----------------------------
chip_w, chip_h = 0.02, 0.015         # dimensoes do chip (2 x 1,5 cm)
chip_x0, chip_y0 = 0.04, 0.0425      # posicao inferior esquerda do chip
P_total = 10.0                       # potencia dissipada (W)
area_chip = chip_w * chip_h
q_chip = P_total / area_chip         # densidade de potencia (W/m²)

# Geracao de calor no dominio
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
q = np.zeros((nx, ny))
mask = (X >= chip_x0) & (X <= chip_x0 + chip_w) & (Y >= chip_y0) & (Y <= chip_y0 + chip_h)
q[mask] = q_chip

# -----------------------------
# 3. Casos simulados
# -----------------------------
cases = [
    ("FR4 baseline", 0.3),      # FR4 puro
    ("Epoxy + 5% FG", 5.0),     # compósito com FG (~5% em massa)
    ("Epoxy + 8% FG", 8.0),     # compósito com FG (~8% em massa)
]


# 4. Funcao de solucao
def solve_steady_state(k, q, max_iter=30000, tol=1e-6):

    """
    Resolve a equacao de Poisson para conducao de calor:
    -k*(d²T/dx² + d²T/dy²) = q
    usando metodo iterativo simples (Gauss-Seidel).
    """
    T = np.full((nx, ny), T_amb, dtype=float)
    for it in range(max_iter):
        T_old = T.copy()
        T[1:-1, 1:-1] = 0.25 * (
            T_old[2:, 1:-1] + T_old[:-2, 1:-1] +
            T_old[1:-1, 2:] + T_old[1:-1, :-2] +
            (dx*dy/k) * q[1:-1, 1:-1]
        )

        # condicoes de contorno fixas
        T[0, :] = T_amb
        T[-1, :] = T_amb
        T[:, 0] = T_amb
        T[:, -1] = T_amb

        # criterio de convergencia
        err = np.max(np.abs(T - T_old))
        if err < tol:
            break
    return T, it+1, err


# 5. Execucao dos casos
for label, k in cases:
    T_map, iters, err = solve_steady_state(k, q)
    Tmax = T_map.max()
    print(f"{label}: k={k} W/mK -> Tmax={Tmax:.2f} °C, ΔT={Tmax - T_amb:.2f} °C, Iterações={iters}")

    # Salvando o mapa termico
    plt.figure(figsize=(6,5))
    plt.imshow(T_map.T, origin='lower', extent=[0, Lx*100, 0, Ly*100])
    plt.imshow(T_map.T, vmin=25, vmax=37, cmap="inferno")
    plt.colorbar(label="Temperatura (°C)")
    plt.title(f"Mapa térmico – {label}")
    plt.xlabel("x (cm)")
    plt.ylabel("y (cm)")
    plt.tight_layout()
    plt.savefig(f"{label.replace(' ', '_')}.png", dpi=200)
    plt.close()

# 6. Graficos comparativos
summary_rows = [
    {"Case": "FR4 baseline", "k_W_mK": 0.3, "Delta_T_C": 11.74},
    {"Case": "Epoxy + 5% FG", "k_W_mK": 5.0, "Delta_T_C": 0.70},
    {"Case": "Epoxy + 8% FG", "k_W_mK": 8.0, "Delta_T_C": 0.44},
]
summary_df = pd.DataFrame(summary_rows)

plt.figure(figsize=(6,4))
bars = plt.bar(summary_df["Case"], summary_df["Delta_T_C"],
               color=["red","orange","green"])
plt.ylabel("ΔT do hotspot (°C)")
plt.title("Comparativo de ΔT do hotspot em função da condutividade térmica")
plt.xticks(rotation=15, ha='right')

# Anotar valores acima das barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2,
             f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("Figura11_BarChart.png", dpi=200)
plt.close()
