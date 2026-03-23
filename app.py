# -*- coding: utf-8 -*-
import json
import streamlit as st
from src import (
    DatiPlatea,
    DEFAULT_STRAT,
    valida_dati,
    calcola_platea,
    tabella_sintesi,
    figura_geometria,
    figura_output,
    figura_cedimenti,
    figura_sezione,
    genera_warning,
    genera_note,
    export_json,
    fem_piastra_winkler,
    figura_momenti_piastra,
    figura_cedimenti_piastra,
    tabella_confronto_platea,
)

DEFAULTS = {
    'B': 10.0,
    'L': 15.0,
    't': 0.8,
    'z_fond': 2.0,
    'N': 15000.0,
    'Mx': 2000.0,
    'My': 1200.0,
    'q_amm': 300.0,
    'kh': 0.15,
    'kv': 0.05,
    'falda': 99.0,
    'profondita_influenza': 5.0,
    'nx': 25,
    'ny': 25,
    'stratigrafia_csv': DEFAULT_STRAT,
}

st.set_page_config(page_title='Platea di fondazione', layout='wide')
st.title('Platea - Statico, sismico e falda')
st.caption('v2.0: sezione verticale, nocciolo, mappe Contour, note tecniche e cedimenti sismici.')

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header('Import / Export input')
    up = st.file_uploader('Reimporta input JSON', type=['json'], key='platea_json')
    defaults = DEFAULTS.copy()
    if up is not None:
        try:
            defaults.update(json.load(up))
            st.success('Input importati.')
        except Exception:
            st.error('JSON non valido.')

    st.header('Geometria')
    B = st.number_input('Dimensione B [m]', 0.5, 100.0, float(defaults['B']), 0.5)
    L = st.number_input('Dimensione L [m]', 0.5, 100.0, float(defaults['L']), 0.5)
    t = st.number_input('Spessore t [m]', 0.1, 10.0, float(defaults['t']), 0.05)
    z_fond = st.number_input('Profondità di posa [m]', 0.0, 50.0, float(defaults['z_fond']), 0.1)
    nx = st.number_input('Punti griglia x [-]', 5, 100, int(defaults['nx']), 1)
    ny = st.number_input('Punti griglia y [-]', 5, 100, int(defaults['ny']), 1)

    st.header('Carichi')
    N = st.number_input('Carico verticale N [kN]', 0.0, 1e7, float(defaults['N']), 100.0)
    Mx = st.number_input('Momento Mx [kNm]', -1e7, 1e7, float(defaults['Mx']), 100.0)
    My = st.number_input('Momento My [kNm]', -1e7, 1e7, float(defaults['My']), 100.0)

    st.header('Sismica e falda')
    kh = st.number_input('kh [-]', 0.0, 1.0, float(defaults['kh']), 0.01)
    kv = st.number_input('kv [-]', 0.0, 1.0, float(defaults['kv']), 0.01)
    falda = st.number_input('Profondità falda [m]', 0.0, 100.0, float(defaults['falda']), 0.1)

    st.header('Terreno')
    q_amm = st.number_input('q ammissibile [kPa]', 50.0, 5000.0, float(defaults['q_amm']), 10.0)
    profondita_influenza = st.number_input(
        'Profondità di influenza [m]', 0.5, 50.0, float(defaults['profondita_influenza']), 0.5
    )

    st.header('Stratigrafia')
    st.caption('Righe: spessore,gamma_dry,gamma_sat,phi,cu,k')
    stratigrafia_csv = st.text_area(
        'Stratigrafia', value=str(defaults['stratigrafia_csv']), height=150
    )

    st.header('FEM Piastra - Parametri')
    E_cls_fem = st.number_input(
        'E calcestruzzo [kPa]', 1e6, 1e8, 3e7, 1e6,
        help='Modulo elastico del calcestruzzo per il calcolo dei momenti nella soletta',
    )
    nu_fem = st.number_input('ν Poisson [-]', 0.1, 0.3, 0.2, 0.01)
    n_grid_fem = st.number_input(
        'Griglia FEM [-]', 4, 30, 14, 1,
        help='Numero di celle per lato. Griglia più fitta = più preciso ma più lento',
    )

# ---------------------------------------------------------------------------
# Calcolo
# ---------------------------------------------------------------------------
d = DatiPlatea(
    B, L, t, z_fond, N, Mx, My, q_amm, kh, kv,
    falda, profondita_influenza, int(nx), int(ny), stratigrafia_csv,
)
err = valida_dati(d)
if err:
    for e in err:
        st.error(e)
    st.stop()

r = calcola_platea(d)
df_sintesi = tabella_sintesi(d, r)
current = {k: v for k, v in d.__dict__.items()}

# Calcolo FEM piastra (opzionale — richiede scipy)
with st.spinner('Calcolo FEM piastra in corso...'):
    fem_statico = fem_piastra_winkler(
        d, seismic=False, E_cls=E_cls_fem, nu=nu_fem, n_grid=int(n_grid_fem)
    )
    fem_sismico = fem_piastra_winkler(
        d, seismic=True, E_cls=E_cls_fem, nu=nu_fem, n_grid=int(n_grid_fem)
    )

# ---------------------------------------------------------------------------
# Warning a livello di pagina
# ---------------------------------------------------------------------------
warnings = genera_warning(d, r)
if warnings:
    with st.expander('Avvertimenti', expanded=True):
        for w in warnings:
            st.warning(w)

# ---------------------------------------------------------------------------
# Metriche (5 colonne)
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('qmax stat. [kPa]', f"{r['statico']['qmax']:.1f}")
c2.metric('qmax sis. [kPa]', f"{r['sismico']['qmax']:.1f}")
c3.metric('ced. stat. [mm]', f"{r['statico']['s_med'] * 1000:.1f}")
c4.metric('ced. sis. [mm]', f"{r['sismico']['s_med'] * 1000:.1f}")
utilizzo_qamm = r['statico']['qmax'] / d.q_amm * 100.0
c5.metric('Utilizzo q_amm [%]', f"{utilizzo_qamm:.1f}")

# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------
t1, t2, t3, t4, t5, t6 = st.tabs([
    'Sintesi',
    'Stratigrafia',
    'Geometria Plotly',
    'Output Plotly',
    'Note tecniche',
    'FEM Piastra',
])

with t1:
    st.dataframe(df_sintesi, use_container_width=True)
    st.download_button(
        'Salva input JSON',
        export_json(current),
        'platea_input.json',
        'application/json',
    )
    st.download_button(
        'Scarica sintesi CSV',
        df_sintesi.to_csv(index=False).encode('utf-8'),
        'platea_sintesi.csv',
        'text/csv',
    )

with t2:
    st.dataframe(r['stratigrafia'], use_container_width=True)

with t3:
    col_geom1, col_geom2 = st.columns(2)
    with col_geom1:
        st.subheader('Pianta')
        st.plotly_chart(figura_geometria(d), use_container_width=True)
    with col_geom2:
        st.subheader('Sezione verticale')
        st.plotly_chart(figura_sezione(d), use_container_width=True)

with t4:
    st.subheader('Pressioni di contatto')
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.plotly_chart(figura_output(r, 'statico'), use_container_width=True)
    with col_p2:
        st.plotly_chart(figura_output(r, 'sismico'), use_container_width=True)

    st.subheader('Cedimenti')
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.plotly_chart(figura_cedimenti(r, 'statico'), use_container_width=True)
    with col_c2:
        st.plotly_chart(figura_cedimenti(r, 'sismico'), use_container_width=True)

with t5:
    st.subheader('Note tecniche e interpretazione dei risultati')
    notes = genera_note(d, r)
    for note in notes:
        st.info(note)

    st.divider()
    st.subheader('Avvertimenti attivi')
    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success('Nessun avvertimento: tutti i controlli soddisfatti.')

    st.divider()
    st.subheader('Parametri di calcolo riepilogati')
    st.markdown(f"""
| Parametro | Valore |
|---|---|
| Dimensioni B × L | {d.B:.2f} m × {d.L:.2f} m |
| Spessore platea t | {d.t:.2f} m |
| Profondità di posa z_fond | {d.z_fond:.2f} m |
| Carico verticale N | {d.N:.0f} kN |
| Momento Mx | {d.Mx:.0f} kNm |
| Momento My | {d.My:.0f} kNm |
| q ammissibile | {d.q_amm:.0f} kPa |
| Coeff. sismici kh / kv | {d.kh:.3f} / {d.kv:.3f} |
| Profondità falda | {d.falda:.2f} m |
| Profondità di influenza | {d.profondita_influenza:.2f} m |
    """)

with t6:
    st.subheader('Analisi strutturale della piastra — Differenze Finite su Winkler')
    st.warning("""
**Ipotesi del modello FEM:**
- Piastra di Kirchhoff (spessa/sottile) con calcestruzzo elastico
- Suolo di Winkler con ks dal modulo di reazione equivalente
- Bordi semplicemente appoggiati (w=0 al bordo) — ipotesi conservativa per i momenti interni
- La distribuzione del carico è quella della soluzione analitica (lineare)
""")
    if fem_statico.get('ok', False):
        st.success(
            f"FEM completato: "
            f"Mx max = {fem_statico['Mx_max']:.1f} kNm/m, "
            f"My max = {fem_statico['My_max']:.1f} kNm/m"
        )
        st.dataframe(
            tabella_confronto_platea(r, fem_statico, fem_sismico),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figura_momenti_piastra(fem_statico, 'Mx'), use_container_width=True)
            st.plotly_chart(figura_momenti_piastra(fem_sismico, 'Mx'), use_container_width=True)
        with col2:
            st.plotly_chart(figura_momenti_piastra(fem_statico, 'My'), use_container_width=True)
            st.plotly_chart(figura_momenti_piastra(fem_sismico, 'My'), use_container_width=True)

        st.plotly_chart(figura_cedimenti_piastra(fem_statico), use_container_width=True)
    else:
        err_msg = fem_statico.get('error', 'scipy non disponibile')
        st.error(f'Calcolo FEM non disponibile: {err_msg}')
        st.info('Installare scipy per abilitare il modulo FEM: pip install scipy')
