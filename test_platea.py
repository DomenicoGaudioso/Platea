# -*- coding: utf-8 -*-
"""
Test del solver Platea — pressioni di contatto Navier, cedimento medio.

Verifica le formule della platea rigida (ipotesi Navier) per pressioni
q_max, q_min e cedimento medio su casi verificabili analiticamente.
Non richiede OpenSeesPy (usa calcola_platea_rigida).
Dati conformi NTC2018, geometrie realistiche.
"""
import math
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src import (
    DatiPlatea,
    calcola_platea_rigida,
    _calcola_carichi_totali,
    riepilogo_platea,
)


def _make_pilastro(P_kN, x, y, Mx_kNm=0.0, My_kNm=0.0):
    """Crea DataFrame con singolo pilastro."""
    return pd.DataFrame(
        [{"x": x, "y": y, "P_kN": P_kN, "Mx_kNm": Mx_kNm, "My_kNm": My_kNm}]
    )


def test_platea_rigida_carico_centrico():
    """Carico centrico N=1200 kN su platea 6×4 m: q_med = N/A = 1200/24 = 50 kPa.

    Con Mx=My=0 la pressione deve essere uniforme: q_max = q_min = 50 kPa.
    Cedimento medio = q_med / k_winkler = 50 / 50000 = 0.001 m = 1 mm.
    """
    pilastri = _make_pilastro(P_kN=1200.0, x=3.0, y=2.0)
    d = DatiPlatea(
        B=6.0, L=4.0, spessore=0.6, E_cls_MPa=30000.0,
        k_winkler_kPa_m=50000.0, mesh_size=0.5,
        pilastri_df=pilastri,
        q_distribuito_kPa=0.0, poisson=0.2,
    )
    r = calcola_platea_rigida(d)
    q = r["pressioni_kPa"]
    q_max = float(np.nanmax(q))
    q_min = float(np.nanmin(q))
    q_attesa = 1200.0 / (6.0 * 4.0)  # = 50 kPa
    assert abs(q_max - q_attesa) < 1.0, f"q_max = {q_max:.2f} kPa, atteso {q_attesa:.2f}"
    assert abs(q_min - q_attesa) < 1.0, f"q_min = {q_min:.2f} kPa, atteso {q_attesa:.2f}"


def test_cedimento_medio_proporzionale():
    """Cedimento medio ∝ pressione media / k_winkler.

    Caso fisico: N=2000 kN, B=5 m, L=5 m, k=40000 kPa/m.
    q_med = 2000/25 = 80 kPa → s_med = 80/40000 = 2 mm.
    """
    pilastri = _make_pilastro(P_kN=2000.0, x=2.5, y=2.5)
    d = DatiPlatea(
        B=5.0, L=5.0, spessore=0.5, E_cls_MPa=30000.0,
        k_winkler_kPa_m=40000.0, mesh_size=0.5,
        pilastri_df=pilastri,
        q_distribuito_kPa=0.0, poisson=0.2,
    )
    r = calcola_platea_rigida(d)
    s_med = float(np.nanmean(r["cedimenti_mm"]))
    q_med = 2000.0 / 25.0  # 80 kPa
    s_atteso_mm = q_med / 40000.0 * 1000.0  # 2 mm
    assert abs(s_med - s_atteso_mm) < 0.2, (
        f"Cedimento medio = {s_med:.3f} mm, atteso {s_atteso_mm:.3f} mm"
    )


def test_platea_rigida_carico_eccentrico_x():
    """Carico eccentrico: N=1000 kN con eccentricità e_x = 1 m su platea 6×4 m.

    q = N/A ± My/Iy × x_rel; My = N × e_x = 1000 kN·m (applicato all'estremo).
    Iy = L × B³/12 = 4×216/12 = 72 m⁴.
    q_max = 50 + 1000×3/72 = 50 + 41.7 = 91.7 kPa.
    q_min = 50 - 41.7 = 8.3 kPa.
    Pilastro a x=4m (baricentro a x=3m → e_x=1m).
    """
    pilastri = _make_pilastro(P_kN=1000.0, x=4.0, y=2.0)
    d = DatiPlatea(
        B=6.0, L=4.0, spessore=0.6, E_cls_MPa=30000.0,
        k_winkler_kPa_m=50000.0, mesh_size=0.5,
        pilastri_df=pilastri,
        q_distribuito_kPa=0.0, poisson=0.2,
    )
    r = calcola_platea_rigida(d)
    q_max = float(np.nanmax(r["pressioni_kPa"]))
    q_min = float(np.nanmin(r["pressioni_kPa"]))

    A = 6.0 * 4.0
    Iy = 4.0 * 6.0**3 / 12.0
    q_med = 1000.0 / A
    My = 1000.0 * 1.0  # N × e_x = 1000 × (4-3) = 1000 kN·m
    q_max_atteso = q_med + My * 3.0 / Iy
    q_min_atteso = q_med - My * 3.0 / Iy

    assert abs(q_max - q_max_atteso) < 2.0, (
        f"q_max = {q_max:.2f} kPa, atteso {q_max_atteso:.2f} kPa"
    )
    assert abs(q_min - q_min_atteso) < 2.0, (
        f"q_min = {q_min:.2f} kPa, atteso {q_min_atteso:.2f} kPa"
    )


def test_calcola_carichi_totali_simmetria():
    """Due pilastri simmetrici: i momenti di eccentricità si cancellano.

    P1=600 kN a x=1.5m, P2=600 kN a x=4.5m su platea 6×3m (baricentro a x=3m).
    N_tot=1200 kN, Mx_ecc=0, My_ecc=0.
    """
    pilastri = pd.DataFrame([
        {"x": 1.5, "y": 1.5, "P_kN": 600.0, "Mx_kNm": 0.0, "My_kNm": 0.0},
        {"x": 4.5, "y": 1.5, "P_kN": 600.0, "Mx_kNm": 0.0, "My_kNm": 0.0},
    ])
    d = DatiPlatea(
        B=6.0, L=3.0, spessore=0.5, E_cls_MPa=30000.0,
        k_winkler_kPa_m=40000.0, mesh_size=0.5,
        pilastri_df=pilastri,
        q_distribuito_kPa=0.0, poisson=0.2,
    )
    N_tot, Mx_tot, My_tot = _calcola_carichi_totali(d)
    assert abs(N_tot - 1200.0) < 0.01, f"N_tot = {N_tot:.1f}, atteso 1200"
    assert abs(My_tot) < 1.0, f"My_tot = {My_tot:.2f}, atteso ≈ 0"
    assert abs(Mx_tot) < 1.0, f"Mx_tot = {Mx_tot:.2f}, atteso ≈ 0"


def test_riepilogo_verifica_geotecnica():
    """Verifica geotecnica: D/C = q_max / q_amm deve essere ≤ 1.0 per verifica OK.

    Caso centrico: q_max = 50 kPa < q_amm = 100 kPa → D/C = 0.5, VERIFICATO.
    """
    pilastri = _make_pilastro(P_kN=1200.0, x=3.0, y=2.0)
    d = DatiPlatea(
        B=6.0, L=4.0, spessore=0.6, E_cls_MPa=30000.0,
        k_winkler_kPa_m=50000.0, mesh_size=0.5,
        pilastri_df=pilastri,
        q_distribuito_kPa=0.0, poisson=0.2,
    )
    r = calcola_platea_rigida(d)
    riepilogo = riepilogo_platea(d, r, q_amm=100.0, nome="Statica")
    assert riepilogo["D_C_pressione"] < 1.0, (
        f"D/C = {riepilogo['D_C_pressione']:.3f}, atteso < 1.0"
    )
    assert riepilogo["Esito"] == "VERIFICATO", (
        f"Esito = {riepilogo['Esito']}, atteso VERIFICATO"
    )


if __name__ == "__main__":
    tests = [
        test_platea_rigida_carico_centrico,
        test_cedimento_medio_proporzionale,
        test_platea_rigida_carico_eccentrico_x,
        test_calcola_carichi_totali_simmetria,
        test_riepilogo_verifica_geotecnica,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} test superati")
    sys.exit(failed)
