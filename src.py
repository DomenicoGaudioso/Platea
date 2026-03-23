# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import asdict
from typing import Tuple, List, Dict
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

# Importazione opzionale di scipy per il modulo FEM piastra
try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

GAMMA_W = 9.81
DEFAULT_STRAT = """2.0,18,20,30,0,25000
3.0,19,21,34,0,40000
5.0,20,20,0,120,60000
"""


def parse_stratigrafia(csv_text: str) -> Tuple[pd.DataFrame, List[str]]:
    """Righe: spessore,gamma_dry,gamma_sat,phi_deg,cu_kPa,k_kN_m3"""
    err, rows = [], []
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame(columns=['spessore_m','gamma_dry','gamma_sat','phi_deg','cu_kPa','k_kN_m3']), ['Inserire almeno uno strato.']
    for i, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
        if len(parts) != 6:
            err.append(f'Riga {i}: usare 6 campi = spessore,gamma_dry,gamma_sat,phi,cu,k.')
            continue
        try:
            h, gd, gs, phi, cu, k = map(float, parts)
            rows.append({'spessore_m': h, 'gamma_dry': gd, 'gamma_sat': gs, 'phi_deg': phi, 'cu_kPa': cu, 'k_kN_m3': k})
        except ValueError:
            err.append(f'Riga {i}: valori non numerici.')
    df = pd.DataFrame(rows)
    if df.empty:
        return df, err or ['Stratigrafia non valida.']
    if (df['spessore_m'] <= 0).any():
        err.append('Tutti gli spessori devono essere positivi.')
    df['z_top_m'] = df['spessore_m'].cumsum() - df['spessore_m']
    df['z_bot_m'] = df['spessore_m'].cumsum()
    return df, err


def layer_at_depth(df: pd.DataFrame, z: float) -> pd.Series:
    sel = df[(df['z_top_m'] <= z) & (df['z_bot_m'] >= z)]
    if sel.empty:
        return df.iloc[-1]
    return sel.iloc[0]


def gamma_eff(layer: pd.Series, z_mid: float, falda_depth: float) -> float:
    if z_mid <= falda_depth:
        return float(layer['gamma_dry'])
    return max(float(layer['gamma_sat']) - GAMMA_W, 1.0)


def sigma_v_eff(df: pd.DataFrame, z: float, falda_depth: float) -> float:
    s = 0.0
    for _, r in df.iterrows():
        a = max(0.0, float(r['z_top_m']))
        b = min(z, float(r['z_bot_m']))
        if b <= a:
            continue
        if falda_depth <= a:
            s += (b - a) * max(float(r['gamma_sat']) - GAMMA_W, 1.0)
        elif falda_depth >= b:
            s += (b - a) * float(r['gamma_dry'])
        else:
            s += (falda_depth - a) * float(r['gamma_dry'])
            s += (b - falda_depth) * max(float(r['gamma_sat']) - GAMMA_W, 1.0)
    return s


def u_hydro(z: float, falda_depth: float) -> float:
    return GAMMA_W * max(z - falda_depth, 0.0)


def export_json(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')


@dataclass(frozen=True)
class DatiPlatea:
    B: float
    L: float
    t: float
    z_fond: float
    N: float
    Mx: float
    My: float
    q_amm: float
    kh: float
    kv: float
    falda: float
    profondita_influenza: float
    nx: int = 25
    ny: int = 25
    stratigrafia_csv: str = DEFAULT_STRAT


def valida_dati(d: DatiPlatea) -> List[str]:
    err = []
    if d.B <= 0 or d.L <= 0 or d.t <= 0:
        err.append('Le dimensioni della platea devono essere positive.')
    if d.q_amm <= 0:
        err.append('q ammissibile deve essere positiva.')
    if d.z_fond < 0:
        err.append('La quota di posa deve essere >= 0.')
    if d.profondita_influenza <= 0:
        err.append('La profondità di influenza deve essere positiva.')
    if d.nx < 5 or d.ny < 5:
        err.append('Usare almeno 5 punti per asse.')
    _, e2 = parse_stratigrafia(d.stratigrafia_csv)
    err.extend(e2)
    return err


def k_equiv(df: pd.DataFrame, Hinf: float) -> float:
    vals = []
    for _, r in df.iterrows():
        a = max(0.0, float(r['z_top_m']))
        b = min(Hinf, float(r['z_bot_m']))
        if b > a:
            vals.append((b - a, float(r['k_kN_m3'])))
    if not vals:
        return 30000.0
    return sum(h * k for h, k in vals) / sum(h for h, _ in vals)


def uplift_pressure_base(z_fond: float, falda: float) -> float:
    return u_hydro(z_fond, falda)


def eval_case(d: DatiPlatea, seismic: bool = False) -> dict:
    df, _ = parse_stratigrafia(d.stratigrafia_csv)
    k_eq = k_equiv(df, d.profondita_influenza)
    A = d.B * d.L
    u0 = uplift_pressure_base(d.z_fond, d.falda)
    Nnet = d.N * (1.0 - (d.kv if seismic else 0.0)) - u0 * A
    Mx = d.Mx * (1.0 + (d.kh if seismic else 0.0))
    My = d.My * (1.0 + (d.kh if seismic else 0.0))
    q0 = Nnet / A
    sx = 6 * My / (d.B * d.L ** 2)
    sy = 6 * Mx / (d.L * d.B ** 2)
    x = np.linspace(-d.B / 2, d.B / 2, d.nx)
    y = np.linspace(-d.L / 2, d.L / 2, d.ny)
    X, Y = np.meshgrid(x, y)
    q = q0 + sx * X + sy * Y
    qmax = float(np.max(q))
    qmin = float(np.min(q))
    s_med = Nnet / max(k_eq * A, 1e-9)
    s_map = q / max(k_eq, 1e-9)
    return {
        'k_eq': k_eq,
        'u_base': u0,
        'Nnet': Nnet,
        'x': x,
        'y': y,
        'q': q,
        'qmax': qmax,
        'qmin': qmin,
        's_med': s_med,
        's_map': s_map,
        'ok': qmax <= d.q_amm and qmin >= 0,
    }


def calcola_platea(d: DatiPlatea) -> Dict[str, object]:
    df, _ = parse_stratigrafia(d.stratigrafia_csv)
    return {'stratigrafia': df, 'statico': eval_case(d, False), 'sismico': eval_case(d, True)}


def tabella_sintesi(d: DatiPlatea, r: Dict[str, object]) -> pd.DataFrame:
    st_r, se = r['statico'], r['sismico']
    rows = [
        ('k equivalente [kN/m³]', st_r['k_eq'], se['k_eq']),
        ('u base [kPa]', st_r['u_base'], se['u_base']),
        ('N netto [kN]', st_r['Nnet'], se['Nnet']),
        ('qmax [kPa]', st_r['qmax'], se['qmax']),
        ('qmin [kPa]', st_r['qmin'], se['qmin']),
        ('cedimento medio [mm]', st_r['s_med'] * 1000, se['s_med'] * 1000),
    ]
    return pd.DataFrame(rows, columns=['Parametro', 'Statico', 'Sismico'])


# ---------------------------------------------------------------------------
# Nuove funzioni: genera_warning e genera_note
# ---------------------------------------------------------------------------

def genera_warning(d: DatiPlatea, r: dict) -> List[str]:
    """Genera messaggi di avvertimento basati sui risultati del calcolo."""
    warns = []
    st_r = r['statico']
    se_r = r['sismico']

    if st_r['qmax'] > d.q_amm:
        warns.append(
            f"⚠ Pressione massima statica ({st_r['qmax']:.1f} kPa) supera q_amm ({d.q_amm:.1f} kPa)"
        )

    if se_r['qmax'] > d.q_amm * 1.3:
        warns.append(
            f"⚠ Pressione massima sismica ({se_r['qmax']:.1f} kPa) supera 1.3×q_amm"
        )

    if st_r['qmin'] < 0:
        warns.append(
            f"⚠ Pressione minima negativa ({st_r['qmin']:.1f} kPa): la fondazione tende a sollevarsi - verifica la zona di trazione"
        )

    if st_r['u_base'] > 0.2 * d.q_amm:
        warns.append(
            f"⚠ Pressione di risalita significativa ({st_r['u_base']:.1f} kPa > 20% q_amm)"
        )

    if st_r['s_med'] * 1000 > 50:
        warns.append(
            f"⚠ Cedimento medio elevato ({st_r['s_med'] * 1000:.1f} mm > 50 mm)"
        )

    ex = abs(d.Mx) / (d.N + 1e-9)
    if ex > d.B / 6:
        warns.append(
            f"⚠ Eccentricità Mx fuori dal nocciolo (e = {ex:.2f} m > B/6 = {d.B / 6:.2f} m)"
        )

    ey = abs(d.My) / (d.N + 1e-9)
    if ey > d.L / 6:
        warns.append(
            f"⚠ Eccentricità My fuori dal nocciolo (e = {ey:.2f} m > L/6 = {d.L / 6:.2f} m)"
        )

    return warns


def genera_note(d: DatiPlatea, r: dict) -> List[str]:
    """Genera note tecniche interpretative sui risultati."""
    notes = []
    st_r = r['statico']
    se_r = r['sismico']
    df, _ = parse_stratigrafia(d.stratigrafia_csv)
    k_eq = k_equiv(df, d.profondita_influenza)

    utilizzo = st_r['qmax'] / d.q_amm * 100.0
    notes.append(f"Utilizzo pressione statica: {utilizzo:.1f}%")

    if se_r['qmax'] > st_r['qmax']:
        notes.append("Il caso sismico governa le pressioni di contatto")

    if st_r['u_base'] > 0:
        notes.append(
            "Uplift idrostatico in fondazione: riduce l'effetto del carico verticale netto"
        )

    if k_eq > 50000:
        notes.append(
            f"Terreno rigido (k_equiv = {k_eq:.0f} kN/m³): cedimenti contenuti"
        )
    elif k_eq < 10000:
        notes.append(
            f"Terreno cedente (k_equiv = {k_eq:.0f} kN/m³): valutare la compatibilità dei cedimenti"
        )

    notes.append("Calcolo del cedimento con modulo di reazione equivalente pesato sugli strati")

    ex = abs(d.Mx) / (d.N + 1e-9)
    if ex <= d.B / 6:
        notes.append(
            f"Eccentricità Mx entro il nocciolo (e = {ex:.2f} m ≤ B/6 = {d.B / 6:.2f} m): nessuna zona di trazione in direzione x"
        )
    else:
        notes.append(
            f"Eccentricità Mx fuori dal nocciolo (e = {ex:.2f} m > B/6 = {d.B / 6:.2f} m): zona di trazione possibile"
        )

    ey = abs(d.My) / (d.N + 1e-9)
    if ey <= d.L / 6:
        notes.append(
            f"Eccentricità My entro il nocciolo (e = {ey:.2f} m ≤ L/6 = {d.L / 6:.2f} m): nessuna zona di trazione in direzione y"
        )
    else:
        notes.append(
            f"Eccentricità My fuori dal nocciolo (e = {ey:.2f} m > L/6 = {d.L / 6:.2f} m): zona di trazione possibile"
        )

    return notes


# ---------------------------------------------------------------------------
# Figura sezione verticale
# ---------------------------------------------------------------------------

def figura_sezione(d: DatiPlatea) -> go.Figure:
    """Sezione verticale della platea con stratigrafia e falda."""
    df, _ = parse_stratigrafia(d.stratigrafia_csv)

    fig = go.Figure()

    # Larghezza del disegno = B della platea
    half_B = d.B / 2.0
    # Profondità massima visualizzata
    z_max = d.z_fond + d.profondita_influenza + 1.0

    # Strati di terreno sotto la quota di posa (da 0 a z_max)
    colori_strati = ['#e8d5b0', '#d4c090', '#c8b07a', '#bca064', '#b09050', '#a4804a']
    for i, (_, row) in enumerate(df.iterrows()):
        z_top_strato = float(row['z_top_m'])
        z_bot_strato = min(float(row['z_bot_m']), z_max)
        if z_bot_strato <= 0:
            continue
        z_top_vis = max(z_top_strato, 0.0)
        colore = colori_strati[i % len(colori_strati)]
        label = f"Strato {i + 1}: φ={row['phi_deg']:.0f}° cu={row['cu_kPa']:.0f} kPa k={row['k_kN_m3']:.0f} kN/m³"
        # Rettangolo strato (in coordinate z positivo verso il basso)
        fig.add_shape(
            type='rect',
            x0=-half_B * 1.5, x1=half_B * 1.5,
            y0=-z_bot_strato, y1=-z_top_vis,
            fillcolor=colore, opacity=0.7,
            line=dict(color='#8B7355', width=0.5),
        )
        # Etichetta strato
        z_mid = -(z_top_vis + z_bot_strato) / 2.0
        fig.add_annotation(
            x=half_B * 1.55, y=z_mid,
            text=label,
            showarrow=False,
            xanchor='left',
            font=dict(size=9, color='#4a3728'),
        )

    # Piano campagna (z=0)
    fig.add_shape(
        type='line',
        x0=-half_B * 2, x1=half_B * 2,
        y0=0, y1=0,
        line=dict(color='#5a8a3c', width=2, dash='solid'),
    )
    fig.add_annotation(
        x=-half_B * 1.9, y=0.15,
        text='Piano campagna',
        showarrow=False,
        font=dict(size=10, color='#5a8a3c'),
    )

    # Terreno sopra la platea (riempimento)
    if d.z_fond > 0:
        fig.add_shape(
            type='rect',
            x0=-half_B, x1=half_B,
            y0=-d.z_fond, y1=0,
            fillcolor='#c8b07a', opacity=0.4,
            line=dict(color='#8B7355', width=0.5),
        )

    # Platea (rettangolo)
    fig.add_shape(
        type='rect',
        x0=-half_B, x1=half_B,
        y0=-d.z_fond - d.t, y1=-d.z_fond,
        fillcolor='#b0b8c8', opacity=0.9,
        line=dict(color='#505870', width=2),
    )
    fig.add_annotation(
        x=0, y=-(d.z_fond + d.t / 2),
        text=f'Platea (t={d.t:.2f} m)',
        showarrow=False,
        font=dict(size=11, color='#303850', family='Arial Black'),
    )

    # Quota di posa
    fig.add_shape(
        type='line',
        x0=-half_B * 1.7, x1=half_B * 1.7,
        y0=-d.z_fond, y1=-d.z_fond,
        line=dict(color='#a04020', width=1.5, dash='dash'),
    )
    fig.add_annotation(
        x=-half_B * 1.65, y=-d.z_fond - 0.15,
        text=f'Quota di posa z={d.z_fond:.2f} m',
        showarrow=False,
        xanchor='left',
        font=dict(size=9, color='#a04020'),
    )

    # Falda (se presente entro z_max)
    if d.falda < z_max:
        fig.add_shape(
            type='line',
            x0=-half_B * 1.7, x1=half_B * 1.7,
            y0=-d.falda, y1=-d.falda,
            line=dict(color='#2060c0', width=2, dash='dot'),
        )
        fig.add_annotation(
            x=-half_B * 1.65, y=-d.falda + 0.15,
            text=f'Falda z={d.falda:.2f} m',
            showarrow=False,
            xanchor='left',
            font=dict(size=9, color='#2060c0'),
        )
        # Simbolo falda
        fig.add_trace(go.Scatter(
            x=np.linspace(-half_B * 1.5, half_B * 1.5, 50),
            y=[-d.falda] * 50,
            mode='lines',
            line=dict(color='rgba(30,100,220,0.25)', width=6),
            name='Falda',
            showlegend=True,
        ))

    # Profondità di influenza
    z_inf = d.z_fond + d.t + d.profondita_influenza
    fig.add_shape(
        type='line',
        x0=-half_B * 1.7, x1=half_B * 1.7,
        y0=-z_inf, y1=-z_inf,
        line=dict(color='#808080', width=1, dash='dashdot'),
    )
    fig.add_annotation(
        x=-half_B * 1.65, y=-z_inf - 0.15,
        text=f'Prof. influenza z={z_inf:.2f} m',
        showarrow=False,
        xanchor='left',
        font=dict(size=9, color='#606060'),
    )

    fig.update_layout(
        title='Sezione verticale della platea',
        xaxis_title='x [m]',
        yaxis_title='Profondità [m] (positivo verso il basso)',
        template='plotly_white',
        showlegend=True,
        height=600,
        margin=dict(r=350),
        xaxis=dict(range=[-half_B * 2, half_B * 2 + d.B * 0.1]),
        yaxis=dict(range=[-(z_max + 0.5), 0.5], autorange=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Figura geometria migliorata (pianta con nocciolo, baricentro, frecce momenti)
# ---------------------------------------------------------------------------

def figura_geometria(d: DatiPlatea) -> go.Figure:
    """Pianta della platea con nocciolo, baricentro e frecce per i momenti."""
    fig = go.Figure()

    # Contorno platea
    xs = [-d.B / 2, d.B / 2, d.B / 2, -d.B / 2, -d.B / 2]
    ys = [-d.L / 2, -d.L / 2, d.L / 2, d.L / 2, -d.L / 2]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        fill='toself',
        fillcolor='rgba(176,184,200,0.4)',
        mode='lines',
        line=dict(color='#505870', width=2),
        name='Platea',
    ))

    # Baricentro
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(symbol='cross', size=14, color='#c03030', line=dict(width=2, color='#c03030')),
        text=['G'],
        textposition='top right',
        textfont=dict(size=12, color='#c03030'),
        name='Baricentro',
    ))

    # Nocciolo (ellisse B/3 x L/3)
    theta = np.linspace(0, 2 * np.pi, 200)
    noc_x = (d.B / 6) * np.cos(theta)
    noc_y = (d.L / 6) * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=noc_x, y=noc_y,
        mode='lines',
        line=dict(color='#e07030', width=2, dash='dash'),
        name=f'Nocciolo (B/3={d.B / 3:.2f}m × L/3={d.L / 3:.2f}m)',
        fill='toself',
        fillcolor='rgba(224,112,48,0.08)',
    ))

    # Eccentricità del carico
    ex = d.Mx / (d.N + 1e-9)
    ey = d.My / (d.N + 1e-9)
    fig.add_trace(go.Scatter(
        x=[ey], y=[ex],
        mode='markers+text',
        marker=dict(symbol='star', size=12, color='#2080c0'),
        text=['N'],
        textposition='top right',
        textfont=dict(size=11, color='#2080c0'),
        name=f'Punto carico (ex={ex:.2f}m, ey={ey:.2f}m)',
    ))

    # Freccia Mx (momento attorno a x → eccentricità in y)
    scala = min(d.B, d.L) * 0.25
    if abs(d.Mx) > 1e-3:
        fig.add_annotation(
            x=0, y=0,
            ax=0, ay=scala * np.sign(d.Mx),
            axref='x', ayref='y',
            xref='x', yref='y',
            arrowhead=3, arrowsize=1.5,
            arrowwidth=2, arrowcolor='#8030c0',
            text='',
        )
        fig.add_annotation(
            x=scala * 0.15, y=scala * np.sign(d.Mx) * 0.6,
            text=f'Mx={d.Mx:.0f} kNm',
            showarrow=False,
            font=dict(size=10, color='#8030c0'),
        )

    # Freccia My (momento attorno a y → eccentricità in x)
    if abs(d.My) > 1e-3:
        fig.add_annotation(
            x=0, y=0,
            ax=scala * np.sign(d.My), ay=0,
            axref='x', ayref='y',
            xref='x', yref='y',
            arrowhead=3, arrowsize=1.5,
            arrowwidth=2, arrowcolor='#30a030',
            text='',
        )
        fig.add_annotation(
            x=scala * np.sign(d.My) * 0.6, y=scala * 0.15,
            text=f'My={d.My:.0f} kNm',
            showarrow=False,
            font=dict(size=10, color='#30a030'),
        )

    # Dimensioni
    fig.add_annotation(
        x=0, y=-d.L / 2 - 0.4,
        text=f'B = {d.B:.2f} m',
        showarrow=False,
        font=dict(size=11),
    )
    fig.add_annotation(
        x=d.B / 2 + 0.4, y=0,
        text=f'L = {d.L:.2f} m',
        showarrow=False,
        font=dict(size=11),
        textangle=-90,
    )

    fig.update_layout(
        title='Geometria della platea - Pianta',
        xaxis_title='x [m]',
        yaxis_title='y [m]',
        template='plotly_white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.3),
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


# ---------------------------------------------------------------------------
# Figura output con go.Contour
# ---------------------------------------------------------------------------

def figura_output(r: Dict[str, object], which: str = 'statico') -> go.Figure:
    """Mappa pressioni di contatto con isolinee (Contour)."""
    rr = r[which]
    fig = go.Figure(data=go.Contour(
        x=rr['x'],
        y=rr['y'],
        z=rr['q'],
        colorbar=dict(title='kPa'),
        colorscale='RdYlGn_r',
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='black'),
        ),
        line_smoothing=0.85,
    ))
    fig.update_layout(
        title=f'Mappa pressioni di contatto - {which}',
        xaxis_title='x [m]',
        yaxis_title='y [m]',
        template='plotly_white',
    )
    return fig


# ---------------------------------------------------------------------------
# Figura cedimenti (invariata)
# ---------------------------------------------------------------------------

def figura_cedimenti(r: Dict[str, object], which: str = 'statico') -> go.Figure:
    rr = r[which]
    fig = go.Figure(data=go.Surface(
        x=rr['x'],
        y=rr['y'],
        z=rr['s_map'] * 1000,
        colorbar=dict(title='mm'),
        colorscale='Blues',
    ))
    fig.update_layout(
        title=f'Superficie cedimenti - {which}',
        scene=dict(
            xaxis_title='x [m]',
            yaxis_title='y [m]',
            zaxis_title='s [mm]',
        ),
        template='plotly_white',
    )
    return fig


# ===========================================================================
# MODULO FEM PIASTRA SU WINKLER — Differenze Finite
# ===========================================================================
# Equazione governante: D*∇⁴w + ks*w = q(x,y)
# dove D = E_cls*t³/(12*(1-ν²))  è la rigidità flessionale [kNm],
# ks è il modulo di Winkler [kN/m³], w il cedimento [m], q la pressione [kPa].
# Condizioni al contorno: bordi semplicemente appoggiati (w=0 al bordo).
# Soluzione numerica con griglia strutturata (Nx+1)×(Ny+1) e assemblaggio
# del sistema sparso tramite scipy.sparse.
# ===========================================================================


def fem_piastra_winkler(d: DatiPlatea, seismic: bool = False,
                         E_cls: float = 30e6, nu: float = 0.2,
                         n_grid: int = 16) -> Dict:
    """
    Soluzione dell'equazione della piastra sottile su fondazione di Winkler
    mediante differenze finite.

    Ipotesi:
    - Piastra di Kirchhoff (sottile) con comportamento elastico lineare
    - Suolo di Winkler con ks dal modulo di reazione equivalente
    - Bordi semplicemente appoggiati (w=0 al bordo) — conservativo per i momenti interni

    Parametri
    ----------
    d        : DatiPlatea
    seismic  : usa caso sismico se True
    E_cls    : modulo elastico calcestruzzo [kPa]
    nu       : coefficiente di Poisson calcestruzzo [-]
    n_grid   : numero di celle in direzione x (default 16); y scalato su L/B

    Ritorna dict con:
    - 'ok'       : bool
    - 'x', 'y'   : vettori coordinata [m]
    - 'w'        : cedimenti [mm], shape (Nx+1, Ny+1)
    - 'Mx','My'  : momenti flettenti [kNm/m], shape (Nx+1, Ny+1)
    - 'Mx_max', 'My_max' : valori massimi assoluti [kNm/m]
    - 'w_max_mm' : cedimento massimo assoluto [mm]
    - 'D_piastra': rigidità flessionale [kNm]
    - 'ks'       : modulo di Winkler equivalente [kN/m³]
    """
    if not _SCIPY_OK:
        return {'error': 'scipy non disponibile', 'ok': False}

    # -----------------------------------------------------------------------
    # Rigidità flessionale della piastra D [kNm]
    # -----------------------------------------------------------------------
    D_piastra = E_cls * d.t ** 3 / (12.0 * (1.0 - nu ** 2))

    # -----------------------------------------------------------------------
    # Griglia: Nx celle in x, Ny celle in y
    # -----------------------------------------------------------------------
    Nx = max(4, min(int(n_grid), 40))
    Ny = max(4, min(int(round(n_grid * d.L / d.B)), 40))
    hx = d.B / Nx   # passo in direzione x [m]
    hy = d.L / Ny   # passo in direzione y [m]

    x = np.linspace(-d.B / 2, d.B / 2, Nx + 1)
    y = np.linspace(-d.L / 2, d.L / 2, Ny + 1)
    # Meshgrid con indexing='ij': X[i,j] = x[i], Y[i,j] = y[j]
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape (Nx+1, Ny+1)

    # -----------------------------------------------------------------------
    # Pressione di carico q(x,y) [kPa] — distribuzione lineare dalla soluzione analitica
    # -----------------------------------------------------------------------
    case_r = eval_case(d, seismic)
    # Ricostruzione della mappa di pressione sulla griglia FEM
    Nnet = case_r['Nnet']
    q0 = Nnet / (d.B * d.L)
    # Gradienti di pressione in direzione x (da My) e y (da Mx)
    My_eff = d.My * (1.0 + (d.kh if seismic else 0.0))
    Mx_eff = d.Mx * (1.0 + (d.kh if seismic else 0.0))
    sx = 6.0 * My_eff / (d.B * d.L ** 2)    # gradiente in x
    sy = 6.0 * Mx_eff / (d.L * d.B ** 2)    # gradiente in y
    Q = q0 + sx * X + sy * Y                 # shape (Nx+1, Ny+1)

    # -----------------------------------------------------------------------
    # Modulo di Winkler ks [kN/m³]
    # -----------------------------------------------------------------------
    df_strat, _ = parse_stratigrafia(d.stratigrafia_csv)
    ks = k_equiv(df_strat, d.profondita_influenza)

    # -----------------------------------------------------------------------
    # Numerazione DOF interni: (i=1..Nx-1, j=1..Ny-1)
    # Nodo (i,j) → DOF interno = (i-1)*(Ny-1) + (j-1)
    # I nodi al bordo (i=0, i=Nx, j=0, j=Ny) hanno w=0 (SS) → non sono incognite
    # -----------------------------------------------------------------------
    N_int = (Nx - 1) * (Ny - 1)  # numero di incognite

    def idx_int(i: int, j: int) -> int:
        """Indice DOF per nodo interno (i=1..Nx-1, j=1..Ny-1)."""
        return (i - 1) * (Ny - 1) + (j - 1)

    # -----------------------------------------------------------------------
    # Assemblaggio matrice sparsa A e vettore b
    # -----------------------------------------------------------------------
    # Coefficienti dello stencil biarmonico per griglia non quadrata hx≠hy:
    #
    #   ∂⁴w/∂x⁴ ≈ (w[i-2,j] - 4w[i-1,j] + 6w[i,j] - 4w[i+1,j] + w[i+2,j]) / hx⁴
    #   ∂⁴w/∂y⁴ ≈ (w[i,j-2] - 4w[i,j-1] + 6w[i,j] - 4w[i,j+1] + w[i,j+2]) / hy⁴
    #   ∂⁴w/∂x²∂y² ≈ (w[i-1,j-1]-2w[i-1,j]+w[i-1,j+1]
    #                  -2(w[i,j-1]-2w[i,j]+w[i,j+1])
    #                  +w[i+1,j-1]-2w[i+1,j]+w[i+1,j+1]) / (hx²*hy²)
    #
    # Coefficienti risultanti moltiplicati per D:
    #   c_center  = D*(6/hx⁴ + 6/hy⁴ + 4/(hx²*hy²)) + ks
    #   c_adj_x   = -D*(4/hx⁴ + 2/(hx²*hy²))   per w[i±1, j]
    #   c_adj_y   = -D*(4/hy⁴ + 2/(hx²*hy²))   per w[i, j±1]
    #   c_far_x   = D/hx⁴                        per w[i±2, j]
    #   c_far_y   = D/hy⁴                        per w[i, j±2]
    #   c_diag    = D*2/(hx²*hy²)                per w[i±1, j±1]

    hx2, hy2 = hx ** 2, hy ** 2
    hx4, hy4 = hx ** 4, hy ** 4
    hx2hy2 = hx2 * hy2

    c_center = D_piastra * (6.0 / hx4 + 6.0 / hy4 + 4.0 / hx2hy2) + ks
    c_adj_x  = -D_piastra * (4.0 / hx4 + 2.0 / hx2hy2)
    c_adj_y  = -D_piastra * (4.0 / hy4 + 2.0 / hx2hy2)
    c_far_x  = D_piastra / hx4
    c_far_y  = D_piastra / hy4
    c_diag   = D_piastra * 2.0 / hx2hy2

    A_mat = lil_matrix((N_int, N_int))
    b_vec = np.zeros(N_int)

    for i in range(1, Nx):
        for j in range(1, Ny):
            row = idx_int(i, j)

            # Termine noto (pressione al nodo)
            b_vec[row] = Q[i, j]

            # Coefficiente diagonale (nodo centrale)
            A_mat[row, row] = c_center

            def add_coeff(i2: int, j2: int, coeff: float) -> None:
                """Aggiunge coeff alla matrice per il nodo (i2,j2).
                Se il nodo è al bordo (w=0), non contribuisce (spostato a RHS=0).
                Se è fuori griglia, non contribuisce (w=0 per estensione).
                """
                if i2 <= 0 or i2 >= Nx or j2 <= 0 or j2 >= Ny:
                    # Bordo o fuori griglia: w=0 → contributo nullo
                    return
                A_mat[row, idx_int(i2, j2)] += coeff

            # Nodi adiacenti in x (distanza hx)
            add_coeff(i - 1, j,     c_adj_x)
            add_coeff(i + 1, j,     c_adj_x)
            # Nodi adiacenti in y (distanza hy)
            add_coeff(i,     j - 1, c_adj_y)
            add_coeff(i,     j + 1, c_adj_y)
            # Nodi lontani in x (distanza 2*hx)
            add_coeff(i - 2, j,     c_far_x)
            add_coeff(i + 2, j,     c_far_x)
            # Nodi lontani in y (distanza 2*hy)
            add_coeff(i,     j - 2, c_far_y)
            add_coeff(i,     j + 2, c_far_y)
            # Nodi diagonali (termine misto ∂⁴/∂x²∂y²)
            add_coeff(i - 1, j - 1, c_diag)
            add_coeff(i - 1, j + 1, c_diag)
            add_coeff(i + 1, j - 1, c_diag)
            add_coeff(i + 1, j + 1, c_diag)

    # -----------------------------------------------------------------------
    # Soluzione del sistema lineare sparso A*w_int = b
    # -----------------------------------------------------------------------
    try:
        w_int = spsolve(A_mat.tocsr(), b_vec)
    except Exception as e:
        return {'error': str(e), 'ok': False}

    # -----------------------------------------------------------------------
    # Ricostruzione campo w completo (Nx+1, Ny+1) con w=0 ai bordi
    # -----------------------------------------------------------------------
    w_full = np.zeros((Nx + 1, Ny + 1))
    for i in range(1, Nx):
        for j in range(1, Ny):
            w_full[i, j] = w_int[idx_int(i, j)]

    # -----------------------------------------------------------------------
    # Momenti flettenti [kNm/m] ai nodi interni
    #
    # Mx = -D*(∂²w/∂x² + ν*∂²w/∂y²)   momento attorno all'asse y (flessione in x)
    # My = -D*(∂²w/∂y² + ν*∂²w/∂x²)   momento attorno all'asse x (flessione in y)
    #
    # Derivate seconde in FD centrate:
    #   ∂²w/∂x²[i,j] = (w[i-1,j] - 2*w[i,j] + w[i+1,j]) / hx²
    #   ∂²w/∂y²[i,j] = (w[i,j-1] - 2*w[i,j] + w[i,j+1]) / hy²
    # -----------------------------------------------------------------------
    Mx_field = np.zeros_like(w_full)
    My_field = np.zeros_like(w_full)
    for i in range(1, Nx):
        for j in range(1, Ny):
            d2x = (w_full[i - 1, j] - 2.0 * w_full[i, j] + w_full[i + 1, j]) / hx2
            d2y = (w_full[i, j - 1] - 2.0 * w_full[i, j] + w_full[i, j + 1]) / hy2
            Mx_field[i, j] = -D_piastra * (d2x + nu * d2y)
            My_field[i, j] = -D_piastra * (d2y + nu * d2x)

    return {
        'ok': True,
        'x': x,
        'y': y,
        'w': w_full * 1000.0,                        # cedimenti in mm
        'Mx': Mx_field,                               # momenti Mx [kNm/m]
        'My': My_field,                               # momenti My [kNm/m]
        'Mx_max': float(np.max(np.abs(Mx_field))),   # massimo assoluto Mx [kNm/m]
        'My_max': float(np.max(np.abs(My_field))),   # massimo assoluto My [kNm/m]
        'w_max_mm': float(np.max(np.abs(w_full)) * 1000.0),
        'D_piastra': D_piastra,
        'ks': ks,
    }


def figura_momenti_piastra(fem_r: Dict, componente: str = 'Mx') -> go.Figure:
    """
    Mappa a contour del momento flettente Mx o My dalla soluzione FEM.

    Colorscale 'RdBu': rosso = momento positivo (trazione nella fibra inferiore),
    blu = momento negativo (trazione nella fibra superiore).

    Parametri
    ----------
    fem_r      : dizionario risultato di fem_piastra_winkler
    componente : 'Mx' o 'My'
    """
    if not fem_r.get('ok', False):
        fig = go.Figure()
        err = fem_r.get('error', 'Risultati FEM non disponibili')
        fig.add_annotation(
            text=f'FEM non disponibile: {err}',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red'),
        )
        fig.update_layout(template='plotly_white')
        return fig

    M = fem_r[componente]        # shape (Nx+1, Ny+1)
    M_max = float(np.max(np.abs(M)))

    # Scala simmetrica attorno a zero per colorscale divergente
    z_lim = max(M_max, 1.0)

    fig = go.Figure(data=go.Contour(
        x=fem_r['x'],
        y=fem_r['y'],
        z=M.T,           # trasposta perché Contour si aspetta z[j,i]
        colorscale='RdBu_r',
        zmin=-z_lim,
        zmax=z_lim,
        colorbar=dict(title='kNm/m'),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=9, color='black'),
        ),
        line_smoothing=0.8,
    ))
    caso = 'Sismico' if fem_r.get('seismic', False) else 'Statico'
    fig.update_layout(
        title=f'Momento {componente} — FEM ({caso}) | max = {M_max:.1f} kNm/m',
        xaxis_title='x [m]',
        yaxis_title='y [m]',
        template='plotly_white',
    )
    return fig


def figura_cedimenti_piastra(fem_r: Dict) -> go.Figure:
    """
    Superficie 3D dei cedimenti dalla soluzione FEM piastra.

    Parametri
    ----------
    fem_r : dizionario risultato di fem_piastra_winkler
    """
    if not fem_r.get('ok', False):
        fig = go.Figure()
        err = fem_r.get('error', 'Risultati FEM non disponibili')
        fig.add_annotation(
            text=f'FEM non disponibile: {err}',
            xref='paper', yref='paper',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='red'),
        )
        fig.update_layout(template='plotly_white')
        return fig

    w_mm = fem_r['w']    # shape (Nx+1, Ny+1), in mm
    w_max = fem_r['w_max_mm']

    fig = go.Figure(data=go.Surface(
        x=fem_r['x'],
        y=fem_r['y'],
        z=w_mm.T,            # trasposta per coerenza con go.Surface (y prima di x)
        colorscale='Blues_r',
        colorbar=dict(title='mm'),
    ))
    fig.update_layout(
        title=f'Cedimenti FEM piastra — w_max = {w_max:.2f} mm',
        scene=dict(
            xaxis_title='x [m]',
            yaxis_title='y [m]',
            zaxis_title='w [mm]',
        ),
        template='plotly_white',
    )
    return fig


def tabella_confronto_platea(r_analitico: Dict, fem_st: Dict, fem_se: Dict) -> 'pd.DataFrame':
    """
    Tabella di confronto tra risultati analitici e FEM per i casi statico e sismico.

    Righe:
    - cedimento medio [mm]: soluzione analitica vs cedimento massimo FEM
    - Mx max [kNm/m]: N/A (non calcolato dall'analitico) vs valore FEM
    - My max [kNm/m]: N/A vs valore FEM

    Parametri
    ----------
    r_analitico : dict da calcola_platea (con chiavi 'statico' e 'sismico')
    fem_st      : dict da fem_piastra_winkler (caso statico)
    fem_se      : dict da fem_piastra_winkler (caso sismico)
    """
    s_med_st = r_analitico['statico']['s_med'] * 1000.0
    s_med_se = r_analitico['sismico']['s_med'] * 1000.0

    def fmt(val, dec=1):
        return f'{val:.{dec}f}' if val is not None else 'N/A'

    rows = [
        (
            'Cedimento medio analitico [mm]',
            fmt(s_med_st),
            fmt(fem_st['w_max_mm'] if fem_st.get('ok') else None),
            fmt(s_med_se),
            fmt(fem_se['w_max_mm'] if fem_se.get('ok') else None),
        ),
        (
            'Mx max [kNm/m]',
            'N/A',
            fmt(fem_st['Mx_max'] if fem_st.get('ok') else None),
            'N/A',
            fmt(fem_se['Mx_max'] if fem_se.get('ok') else None),
        ),
        (
            'My max [kNm/m]',
            'N/A',
            fmt(fem_st['My_max'] if fem_st.get('ok') else None),
            'N/A',
            fmt(fem_se['My_max'] if fem_se.get('ok') else None),
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=['Parametro', 'Analitico Statico', 'FEM Statico', 'Analitico Sismico', 'FEM Sismico'],
    )
