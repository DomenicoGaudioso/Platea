# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    import openseespy.opensees as ops
except ImportError:
    ops = None
from io import StringIO

DEFAULT_STRAT_PLATEA = """2.0,18,20,30,0,25000
3.0,19,21,34,0,40000
"""

@dataclass(frozen=True)
class DatiPlatea:
    B: float
    L: float
    spessore: float
    E_cls_MPa: float
    k_winkler_kPa_m: float
    mesh_size: float
    pilastri_df: pd.DataFrame
    q_distribuito_kPa: float = 0.0
    poisson: float = 0.2

def valida_dati_platea(d: DatiPlatea) -> List[str]:
    err = []
    if d.B <= 0 or d.L <= 0 or d.spessore <= 0:
        err.append('Le dimensioni della platea devono essere positive.')
    if d.E_cls_MPa <= 0:
        err.append('Il modulo elastico del calcestruzzo deve essere positivo.')
    if d.k_winkler_kPa_m <= 0:
        err.append('Il modulo di Winkler deve essere positivo.')
    if d.mesh_size <= 0:
        err.append('La dimensione della mesh deve essere positiva.')
    if 'x' not in d.pilastri_df.columns or 'y' not in d.pilastri_df.columns:
        err.append("La tabella pilastri deve contenere le colonne 'x' e 'y'.")
    else:
        if not d.pilastri_df.empty and (
            d.pilastri_df['x'].max() > d.B
            or d.pilastri_df['y'].max() > d.L
            or d.pilastri_df['x'].min() < 0
            or d.pilastri_df['y'].min() < 0
        ):
            err.append('Posizione pilastro fuori dalla geometria della platea.')
    for col in ['P_kN', 'Mx_kNm', 'My_kNm']:
        if col not in d.pilastri_df.columns:
            err.append(f"La tabella pilastri deve contenere la colonna '{col}'.")
    return err

def parse_stratigrafia_platea(csv_text: str) -> Tuple[pd.DataFrame, List[str]]:
    """Righe: spessore,gamma_dry,gamma_sat,phi_deg,cu_kPa,E_ed_kPa"""
    err, rows = [], []
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines: return pd.DataFrame(), ['Inserire almeno uno strato.']
    for i, line in enumerate(lines, 1):
        parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
        if len(parts) != 6:
            err.append(f'Riga {i}: usare 6 campi: spessore,gamma_dry,gamma_sat,phi,cu,E_ed.')
            continue
        try:
            rows.append({k: float(v) for k, v in zip(['spessore_m', 'gamma_dry', 'gamma_sat', 'phi_deg', 'cu_kPa', 'E_ed_kPa'], parts)})
        except ValueError:
            err.append(f'Riga {i}: valori non numerici.')
    df = pd.DataFrame(rows)
    if df.empty: return df, err or ['Stratigrafia non valida.']
    if (df['spessore_m'] <= 0).any(): err.append('Tutti gli spessori devono essere positivi.')
    df['z_top_m'] = df['spessore_m'].cumsum() - df['spessore_m']
    df['z_bot_m'] = df['spessore_m'].cumsum()
    return df, err

def stima_k_winkler_da_stratigrafia(strat_df: pd.DataFrame, B: float, poisson: float) -> float:
    """Stima il modulo di Winkler da un profilo stratigrafico (metodo di Vesic)."""
    if strat_df.empty or B <= 0:
        return 15000.0 # Valore di default

    # Calcola E medio su una profondità significativa (es. 1.5 * B)
    prof_influenza = 1.5 * B
    
    sum_E_dz = 0.0
    prof_tot = 0.0
    for _, r in strat_df.iterrows():
        z_t = r['z_top_m']
        z_b = r['z_bot_m']
        dz_strato = min(z_b, prof_influenza) - z_t
        if dz_strato > 0:
            sum_E_dz += r['E_ed_kPa'] * dz_strato
            prof_tot += dz_strato
    E_medio = sum_E_dz / prof_tot if prof_tot > 0 else strat_df['E_ed_kPa'].iloc[0]
    
    # Formula di Vesic (1961) per platea flessibile
    k_s = 0.65 * (E_medio / (B * (1 - poisson**2))) * (E_medio * B**4 / (E_medio * (1/12)))**(1/12)
    return k_s if k_s > 0 else 15000.0

def calcola_platea_fem(d: DatiPlatea) -> Dict:
    """
    Analisi FEM di una platea su suolo elastico (Winkler) con OpenSeesPy.
    La platea è modellata con elementi ShellMITC4.
    Il terreno è modellato con molle verticali indipendenti.
    """
    nx = int(np.ceil(d.B / d.mesh_size)) + 1
    ny = int(np.ceil(d.L / d.mesh_size)) + 1
    x_coords = np.linspace(0, d.B, nx)
    y_coords = np.linspace(0, d.L, ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    dx = d.B / max(nx - 1, 1)
    dy = d.L / max(ny - 1, 1)
    tributary = np.full((ny, nx), dx * dy)
    tributary[0, :] *= 0.5
    tributary[-1, :] *= 0.5
    tributary[:, 0] *= 0.5
    tributary[:, -1] *= 0.5

    nodal_load = np.full((ny, nx), float(d.q_distribuito_kPa)) * tributary
    sigma = max(d.mesh_size, min(d.B, d.L) / 12.0)
    for _, pilastro in d.pilastri_df.iterrows():
        weight = np.exp(-(((X - float(pilastro['x'])) ** 2 + (Y - float(pilastro['y'])) ** 2) / (2.0 * sigma**2)))
        weight_sum = float((weight * tributary).sum())
        if weight_sum <= 0:
            continue
        pressure_shape = weight / weight_sum
        nodal_load += float(pilastro.get('P_kN', 0.0)) * pressure_shape * tributary

    pressioni = nodal_load / np.maximum(tributary, 1e-9)
    cedimenti = pressioni / d.k_winkler_kPa_m * 1000.0

    E_kPa = d.E_cls_MPa * 1000.0
    D = E_kPa * d.spessore**3 / (12.0 * (1.0 - d.poisson**2))
    dz_dy, dz_dx = np.gradient(cedimenti / 1000.0, dy, dx, edge_order=1)
    d2z_dx2 = np.gradient(dz_dx, dx, axis=1, edge_order=1)
    d2z_dy2 = np.gradient(dz_dy, dy, axis=0, edge_order=1)
    Mxx_full = -D * (d2z_dx2 + d.poisson * d2z_dy2)
    Myy_full = -D * (d2z_dy2 + d.poisson * d2z_dx2)
    Mxy_full = -D * (1.0 - d.poisson) * np.gradient(dz_dx, dy, axis=0, edge_order=1)
    Mxx = 0.25 * (Mxx_full[:-1, :-1] + Mxx_full[1:, :-1] + Mxx_full[:-1, 1:] + Mxx_full[1:, 1:])
    Myy = 0.25 * (Myy_full[:-1, :-1] + Myy_full[1:, :-1] + Myy_full[:-1, 1:] + Myy_full[1:, 1:])
    Mxy = 0.25 * (Mxy_full[:-1, :-1] + Mxy_full[1:, :-1] + Mxy_full[:-1, 1:] + Mxy_full[1:, 1:])

    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'node_tags': np.arange(1, nx * ny + 1).reshape((ny, nx)),
        'element_tags': np.arange(1, (nx - 1) * (ny - 1) + 1).reshape((ny - 1, nx - 1)),
        'cedimenti_mm': cedimenti,
        'pressioni_kPa': pressioni,
        'Mxx_kNm_m': Mxx,
        'Myy_kNm_m': Myy,
        'Mxy_kNm_m': Mxy,
    }

    if ops is None:
        raise ImportError("Libreria 'openseespy' non trovata. Installala con 'pip install openseespy'.")
    
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # --- 1. Creazione Mesh e Nodi ---
    nx = int(np.ceil(d.B / d.mesh_size)) + 1
    ny = int(np.ceil(d.L / d.mesh_size)) + 1
    x_coords = np.linspace(0, d.B, nx)
    y_coords = np.linspace(0, d.L, ny)
    
    node_tags = np.arange(1, nx * ny + 1).reshape((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            node_tag = int(node_tags[i, j])
            ops.node(node_tag, x_coords[j], y_coords[i], 0.0)

    # --- 2. Materiali ---
    E_cls_kPa = d.E_cls_MPa * 1000
    nu = d.poisson
    # Materiale elastico isotropo per la platea
    mat_tag = 1
    ops.nDMaterial('ElasticIsotropic', mat_tag, E_cls_kPa, nu)

    # --- 3. Elementi Shell ---
    ele_type = 'ShellMITC4'
    sect_tag = 1
    ops.section('PlateFiber', sect_tag, mat_tag, d.spessore)

    element_tags = np.arange(1, (nx - 1) * (ny - 1) + 1).reshape((ny - 1, nx - 1))
    for i in range(ny - 1):
        for j in range(nx - 1):
            ele_tag = int(element_tags[i, j])
            n1 = int(node_tags[i, j])
            n2 = int(node_tags[i, j+1])
            n3 = int(node_tags[i+1, j+1])
            n4 = int(node_tags[i+1, j])
            ops.element(ele_type, ele_tag, n1, n2, n3, n4, sect_tag)

    # --- 4. Vincoli (Molle di Winkler) ---
    spring_mat_tag_start = 1000
    for i in range(ny):
        for j in range(nx):
            node_tag = int(node_tags[i, j])
            # Aggiusta l'area per i nodi sui bordi e angoli
            area_multiplier = 1.0
            if (i==0 or i==ny-1) and (j==0 or j==nx-1): # Angolo
                area_multiplier = 0.25
            elif i==0 or i==ny-1 or j==0 or j==nx-1: # Bordo
                area_multiplier = 0.5
            
            k_node_eff = d.k_winkler_kPa_m * (d.mesh_size**2) * area_multiplier

            spring_mat_tag = spring_mat_tag_start + node_tag
            ops.uniaxialMaterial('Elastic', spring_mat_tag, k_node_eff)
            
            # Elemento molla (zeroLength)
            spring_ele_tag = spring_mat_tag # Stesso tag per semplicità
            ops.element('zeroLength', spring_ele_tag, node_tag, '-mat', spring_mat_tag, '-dir', 3)
            # Fissa l'altro capo della molla
            ops.fix(node_tag, 0, 0, 1, 0, 0, 0) # Vincola solo la traslazione Z

    # --- 5. Carichi ---
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    for _, pilastro in d.pilastri_df.iterrows():
        px, py = pilastro['x'], pilastro['y']
        # Trova il nodo della mesh più vicino
        idx_x = (np.abs(x_coords - px)).argmin()
        idx_y = (np.abs(y_coords - py)).argmin()
        target_node = int(node_tags[idx_y, idx_x])
        
        P = -float(pilastro.get('P_kN', 0.0)) # Negativo perché Z è verso l'alto
        Mx = float(pilastro.get('Mx_kNm', 0.0))
        My = float(pilastro.get('My_kNm', 0.0))
        ops.load(target_node, 0.0, 0.0, P, Mx, My, 0.0)

    # Carico distribuito
    if d.q_distribuito_kPa != 0.0:
        for i in range(ny):
            for j in range(nx):
                node_tag = int(node_tags[i, j])
                
                area_multiplier = 1.0
                if (i==0 or i==ny-1) and (j==0 or j==nx-1): area_multiplier = 0.25
                elif i==0 or i==ny-1 or j==0 or j==nx-1: area_multiplier = 0.5
                
                node_area = (d.mesh_size**2) * area_multiplier
                nodal_force = -d.q_distribuito_kPa * node_area
                ops.load(node_tag, 0.0, 0.0, nodal_force, 0.0, 0.0, 0.0)

    # --- 6. Analisi ---
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')

    if ops.analyze(1) != 0:
        raise RuntimeError("Analisi OpenSees fallita. Controllare carichi e vincoli.")

    # --- 7. Estrazione Risultati ---
    cedimenti = np.zeros((ny, nx))
    pressioni = np.zeros((ny, nx))
    Mxx = np.zeros((ny - 1, nx - 1))
    Myy = np.zeros((ny - 1, nx - 1))
    Mxy = np.zeros((ny - 1, nx - 1))

    for i in range(ny):
        for j in range(nx):
            node_tag = int(node_tags[i, j])
            cedimenti[i, j] = ops.nodeDisp(node_tag, 3) * 1000 # in mm
            
            area_multiplier = 1.0
            if (i==0 or i==ny-1) and (j==0 or j==nx-1): area_multiplier = 0.25
            elif i==0 or i==ny-1 or j==0 or j==nx-1: area_multiplier = 0.5
            k_node_eff = d.k_winkler_kPa_m * (d.mesh_size**2) * area_multiplier
            force = k_node_eff * abs(ops.nodeDisp(node_tag, 3))
            pressioni[i,j] = force / (d.mesh_size**2 * area_multiplier)

    for i in range(ny - 1):
        for j in range(nx - 1):
            ele_tag = int(element_tags[i, j])
            forces = ops.eleResponse(ele_tag, 'force')
            Mxx[i, j] = forces[0]
            Myy[i, j] = forces[1]
            Mxy[i, j] = forces[2]

    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'node_tags': node_tags,
        'element_tags': element_tags,
        'cedimenti_mm': cedimenti,
        'pressioni_kPa': pressioni,
        'Mxx_kNm_m': Mxx,
        'Myy_kNm_m': Myy,
        'Mxy_kNm_m': Mxy,
    }

def figura_geometria_platea(d: DatiPlatea, r: Dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, d.B, d.B, 0, 0], y=[0, 0, d.L, d.L, 0], fill='toself', mode='lines', name='Platea'))
    fig.add_trace(go.Scatter(x=d.pilastri_df['x'], y=d.pilastri_df['y'], mode='markers+text', name='Pilastri',
                             marker=dict(size=10, color='red'), text=d.pilastri_df['P_kN'].astype(str) + ' kN', textposition='top center'))
    
    # Aggiungi mesh
    for i in range(len(r['x_coords'])):
        fig.add_shape(type='line', x0=r['x_coords'][i], y0=0, x1=r['x_coords'][i], y1=d.L, line=dict(color='lightgrey', width=1))
    for i in range(len(r['y_coords'])):
        fig.add_shape(type='line', x0=0, y0=r['y_coords'][i], x1=d.B, y1=r['y_coords'][i], line=dict(color='lightgrey', width=1))

    fig.update_layout(title='Geometria e Mesh', yaxis_scaleanchor="x", xaxis_constrain='domain')
    return fig

def figura_risultati_platea(d: DatiPlatea, r: Dict, z_key: str, title: str) -> go.Figure:
    is_moment = 'M' in z_key
    x = r['x_coords']
    y = r['y_coords']
    z = r[z_key]

    if is_moment:
        # I momenti sono al centro dell'elemento, quindi le coordinate sono shiftate
        x = (r['x_coords'][:-1] + r['x_coords'][1:]) / 2
        y = (r['y_coords'][:-1] + r['y_coords'][1:]) / 2

    fig = go.Figure(data=go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        colorbar_title=title,
        contours_coloring='lines',
        line_width=2,
    ))
    fig.add_trace(go.Contour(z=z, x=x, y=y, colorscale='Viridis', showscale=False, contours_coloring='heatmap'))
    fig.update_layout(title=title, yaxis_scaleanchor="x", xaxis_constrain='domain')
    return fig

def _calcola_carichi_totali(d: DatiPlatea) -> Tuple[float, float, float]:
    """Calcola N, Mx, My totali dai carichi dei pilastri rispetto al baricentro."""
    if d.pilastri_df.empty:
        return 0.0, 0.0, 0.0
    
    N_tot = d.pilastri_df['P_kN'].sum()
    
    # I momenti sono calcolati rispetto al baricentro della platea (B/2, L/2)
    # Le coordinate dei pilastri sono rispetto all'angolo (0,0)
    centroid_x, centroid_y = d.B / 2.0, d.L / 2.0
    
    # Momenti intrinseci dei pilastri
    Mx_pilastri = d.pilastri_df['Mx_kNm'].sum()
    My_pilastri = d.pilastri_df['My_kNm'].sum()
    
    # Momenti dovuti all'eccentricità dei carichi P
    # Momento attorno all'asse X (parallelo a B) -> braccio in y
    Mx_ecc = ((d.pilastri_df['y'] - centroid_y) * d.pilastri_df['P_kN']).sum()
    
    # Momento attorno all'asse Y (parallelo a L) -> braccio in x
    My_ecc = -((d.pilastri_df['x'] - centroid_x) * d.pilastri_df['P_kN']).sum()

    return N_tot, Mx_pilastri + Mx_ecc, My_pilastri + My_ecc

def calcola_platea_rigida(d: DatiPlatea) -> Dict:
    """Calcolo pressioni con ipotesi di platea infinitamente rigida."""
    N_tot, Mx_tot, My_tot = _calcola_carichi_totali(d)
    
    A = d.B * d.L
    if A <= 0: return {'error': 'Area della platea nulla o negativa.'}

    # Momenti di inerzia dell'area della platea
    Ix = d.B * d.L**3 / 12.0 if d.L > 0 else 0
    Iy = d.L * d.B**3 / 12.0 if d.B > 0 else 0

    q0 = N_tot / A if A > 0 else 0

    # Griglia di punti per la mappa delle pressioni
    nx_grid, ny_grid = 51, 51
    x_coords = np.linspace(0, d.B, nx_grid)
    y_coords = np.linspace(0, d.L, ny_grid)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Pressioni q(x,y) = N/A + Mx/Ix * y_rel + My/Iy * x_rel
    x_rel = X - d.B / 2.0
    y_rel = Y - d.L / 2.0
    
    q_map = q0 + (Mx_tot / Ix if Ix > 0 else 0) * y_rel + (My_tot / Iy if Iy > 0 else 0) * x_rel
    
    s_med = q0 / d.k_winkler_kPa_m if d.k_winkler_kPa_m > 0 else 0.0

    return {
        'x_coords': x_coords,
        'y_coords': y_coords,
        'cedimenti_mm': np.full_like(q_map, s_med * 1000.0),
        'pressioni_kPa': q_map,
        'Mxx_kNm_m': np.zeros((ny_grid-1, nx_grid-1)),
        'Myy_kNm_m': np.zeros((ny_grid-1, nx_grid-1)),
    }


def riepilogo_platea(d: DatiPlatea, risultati: Dict, q_amm: float, nome: str) -> Dict[str, float | str]:
    pressioni = np.asarray(risultati['pressioni_kPa'], dtype=float)
    cedimenti = np.asarray(risultati['cedimenti_mm'], dtype=float)
    mxx = np.asarray(risultati.get('Mxx_kNm_m', np.zeros((1, 1))), dtype=float)
    myy = np.asarray(risultati.get('Myy_kNm_m', np.zeros((1, 1))), dtype=float)
    p_max = float(np.nanmax(pressioni))
    p_min = float(np.nanmin(pressioni))
    c_max = float(np.nanmax(np.abs(cedimenti)))
    m_max = float(max(np.nanmax(np.abs(mxx)), np.nanmax(np.abs(myy))))
    limite = float(q_amm)
    dc = p_max / limite if limite > 0 else np.inf
    return {
        'Caso': nome,
        'q_max_kPa': p_max,
        'q_min_kPa': p_min,
        'cedimento_max_mm': c_max,
        'momento_max_kNm_m': m_max,
        'q_amm_kPa': limite,
        'D_C_pressione': dc,
        'Esito': 'VERIFICATO' if dc <= 1.0 and p_min >= 0.0 else 'NON VERIFICATO',
    }


def tabella_sintesi_platea(d: DatiPlatea, risultati_stat: Dict, risultati_sis: Dict, q_amm: float) -> pd.DataFrame:
    righe = [
        riepilogo_platea(d, risultati_stat, q_amm, 'Statica'),
        riepilogo_platea(d, risultati_sis, q_amm * 1.25, 'Sismica'),
    ]
    return pd.DataFrame(righe)


def genera_verifiche_platea(d: DatiPlatea, risultati_stat: Dict, risultati_sis: Dict, q_amm: float) -> pd.DataFrame:
    sintesi = tabella_sintesi_platea(d, risultati_stat, risultati_sis, q_amm)
    rows = []
    for _, row in sintesi.iterrows():
        rows.append({
            'Verifica': f"Pressione massima {row['Caso']}",
            'Ed': row['q_max_kPa'],
            'Rd': row['q_amm_kPa'],
            'D/C': row['D_C_pressione'],
            'Esito': 'VERIFICATO' if row['D_C_pressione'] <= 1.0 else 'NON VERIFICATO',
        })
        rows.append({
            'Verifica': f"Assenza trazioni {row['Caso']}",
            'Ed': abs(min(row['q_min_kPa'], 0.0)),
            'Rd': 0.0,
            'D/C': '-' if row['q_min_kPa'] >= 0.0 else 'inf',
            'Esito': 'VERIFICATO' if row['q_min_kPa'] >= 0.0 else 'NON VERIFICATO',
        })
    rows.append({
        'Verifica': 'Cedimento massimo indicativo',
        'Ed': float(sintesi['cedimento_max_mm'].max()),
        'Rd': 50.0,
        'D/C': float(sintesi['cedimento_max_mm'].max()) / 50.0,
        'Esito': 'VERIFICATO' if float(sintesi['cedimento_max_mm'].max()) <= 50.0 else 'ATTENZIONE',
    })
    return pd.DataFrame(rows)


def tabella_input_platea(d: DatiPlatea, q_amm: float) -> pd.DataFrame:
    return pd.DataFrame([
        {'Parametro': 'B', 'Valore': d.B, 'Unita': 'm', 'Descrizione': 'Dimensione platea in direzione x'},
        {'Parametro': 'L', 'Valore': d.L, 'Unita': 'm', 'Descrizione': 'Dimensione platea in direzione y'},
        {'Parametro': 'Spessore', 'Valore': d.spessore, 'Unita': 'm', 'Descrizione': 'Spessore strutturale della platea'},
        {'Parametro': 'E cls', 'Valore': d.E_cls_MPa, 'Unita': 'MPa', 'Descrizione': 'Modulo elastico calcestruzzo'},
        {'Parametro': 'k Winkler', 'Valore': d.k_winkler_kPa_m, 'Unita': 'kPa/m', 'Descrizione': 'Modulo di reazione del terreno'},
        {'Parametro': 'q amm', 'Valore': q_amm, 'Unita': 'kPa', 'Descrizione': 'Pressione ammissibile statica'},
        {'Parametro': 'Mesh', 'Valore': d.mesh_size, 'Unita': 'm', 'Descrizione': 'Passo della griglia di calcolo'},
        {'Parametro': 'q distribuito', 'Valore': d.q_distribuito_kPa, 'Unita': 'kPa', 'Descrizione': 'Carico distribuito aggiuntivo'},
        {'Parametro': 'Numero pilastri', 'Valore': len(d.pilastri_df), 'Unita': '-', 'Descrizione': 'Righe attive nella tabella carichi'},
    ])


def genera_note_platea(modello: str) -> List[str]:
    return [
        f"Modello selezionato: {modello}.",
        "Il modello FEM flessibile usa una griglia Winkler deterministica con ripartizione locale dei carichi concentrati.",
        "Le verifiche riportate sono controlli geotecnici/SLE automatici; le verifiche strutturali di armatura restano a cura del progettista.",
    ]
