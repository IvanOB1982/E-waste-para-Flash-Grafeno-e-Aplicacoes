"""
Apendice A – Simulacao de retencao de carga em baterias
Autor: Ivan de Oliveira Bessa RU: 3613650
Curso: Engenharia da Computacao
Instituicao: UNINTER
Ano: 2025

Descricao:
-----------
Este codigo implementa uma simulacao da dopagem de anodos de baterias de ions de litio por
Flash grafeno.
O objetivo eh avaliar o efeito da carga e descarga ao longo de 500 ciclos, confiabilidade
e eficiencia energetica
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Configuracao dos cenarios
# Cada cenario define capacidade inicial (C0, mAh/g) e retencao desejada aos 500 ciclos (fracao).
SCENARIOS = [
    {"label": "Grafite (baseline)", "C0": 370.0,  "retention_500": 0.80},
    {"label": "Grafite + 5% FG",    "C0": 420.0,  "retention_500": 0.92},
    {"label": "Si + 10% FG",        "C0": 1500.0, "retention_500": 0.85},
]

N_CYCLES = 500
OUTDIR = Path("resultados")   # pasta de saida (sera criada)


# Modelo de degradação
# Usamos um decaimento exponencial simples:
#   C(n) = C0 * exp(-k * n)
# com k escolhido para garantir C(500)/C0 = retention_500  ->  k = -ln(r)/500
def simulate_series(scenarios, n_cycles):
    cycles = np.arange(0, n_cycles + 1, 1)
    series = {}
    enriched = []
    for s in scenarios:
        r = float(s["retention_500"])
        k = -np.log(r) / n_cycles
        C0 = float(s["C0"])
        Cn = C0 * np.exp(-k * cycles)
        s_enr = dict(s)
        s_enr["k"] = k
        s_enr["C_500"] = float(Cn[-1])
        series[s["label"]] = Cn
        enriched.append(s_enr)
    return cycles, series, enriched


# Geracao dos graficos
def plot_capacity_vs_cycles(cycles, series, outpath: Path):
    plt.figure(figsize=(7, 4.5))
    for label, Cn in series.items():
        plt.plot(cycles, Cn, label=label)
    plt.xlabel("Ciclos")
    plt.ylabel("Capacidade específica (mAh/g)")
    plt.title("Figura 12 – Capacidade × Número de ciclos (modelo exponencial)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_retained_capacity(enriched, outpath: Path):
    labels = [s["label"] for s in enriched]
    cap_500 = [s["C_500"] for s in enriched]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, cap_500)
    plt.ylabel("Capacidade após 500 ciclos (mAh/g)")
    plt.title("Figura 13 – Capacidade retida após 500 ciclos")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def save_summary_csv(enriched, outpath: Path):
    df = pd.DataFrame([
        {
            "Caso": s["label"],
            "C0_mAh_g": s["C0"],
            "ret_500_frac": s["retention_500"],
            "k_decay_per_cycle": s["k"],
            "C_500_mAh_g": s["C_500"],
        } for s in enriched
    ])
    df.to_csv(outpath, index=False)
    return df


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cycles, series, enriched = simulate_series(SCENARIOS, N_CYCLES)

    fig12_path = OUTDIR / "Figure12_Capacity_vs_Cycles.png"
    fig13_path = OUTDIR / "Figure13_Capacity_Retention_Bars.png"
    csv_path  = OUTDIR / "Figure12_13_Battery_Summary.csv"

    plot_capacity_vs_cycles(cycles, series, fig12_path)
    plot_retained_capacity(enriched, fig13_path)
    df = save_summary_csv(enriched, csv_path)

    # Relato simples no console
    print(f"[OK] Figura 12 salva em: {fig12_path}")
    print(f"[OK] Figura 13 salva em: {fig13_path}")
    print(f"[OK] CSV salvo em:      {csv_path}\n")
    print(df)

if __name__ == "__main__":
    main()
