# -*- coding: utf-8 -*-
import json
import pandas as pd
import streamlit as st
from src import (
    DatiPlatea,
    DEFAULT_STRAT_PLATEA,
    calcola_platea_rigida,
    valida_dati_platea,
    parse_stratigrafia_platea,
    calcola_platea_fem,
    figura_geometria_platea,
    figura_risultati_platea,
    genera_note_platea,
    genera_verifiche_platea,
    stima_k_winkler_da_stratigrafia,
    tabella_sintesi_platea,
)

# Prova a importare le librerie per il reporting
try:
    from report import create_word_report
    reporting_enabled = True
except ImportError:
    reporting_enabled = False

PREMIUM_CSS = """
<style>
  :root {
    --ink: #13202b;
    --muted: #5b6774;
    --line: rgba(19, 32, 43, 0.12);
    --paper: #f6f7f8;
    --panel: #ffffff;
    --accent: #155e75;
    --ok: #166534;
    --warn: #a16207;
    --bad: #b91c1c;
  }
  .stApp {
    background: linear-gradient(180deg, #f7f8fa 0%, #eef2f5 100%);
  }
  .block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
  }
  .hero-shell {
    border: 1px solid rgba(255,255,255,.22);
    background: linear-gradient(135deg, #13202b 0%, #164e63 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 14px;
    box-shadow: 0 18px 42px rgba(19, 32, 43, .16);
    margin-bottom: 1rem;
  }
  .hero-shell h1 {
    margin: 0 0 .35rem;
    font-size: 2.05rem;
    line-height: 1.05;
    letter-spacing: 0;
  }
  .hero-shell p {
    margin: 0;
    max-width: 860px;
    color: rgba(255,255,255,.82);
    font-size: 1.02rem;
  }
  .hero-badges {
    display: flex;
    gap: .5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
  }
  .hero-badge {
    border: 1px solid rgba(255,255,255,.18);
    background: rgba(255,255,255,.10);
    border-radius: 8px;
    padding: .28rem .72rem;
    font-size: .78rem;
  }
  .section-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    box-shadow: 0 8px 24px rgba(20, 39, 55, .05);
    margin-bottom: .85rem;
    color: var(--ink);
  }
  .section-card:empty {
    display: none;
  }
  .section-card h1,
  .section-card h2,
  .section-card h3,
  .section-card p,
  .section-card li,
  .section-card span {
    color: var(--ink);
  }
  div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: .8rem 1rem;
    box-shadow: 0 8px 20px rgba(20, 39, 55, .04);
  }
  div[data-testid="stMetric"] * {
    color: var(--ink) !important;
  }
  div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
  }
  div[data-testid="stMetricValue"] {
    color: var(--ink) !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.62);
    border: 1px solid var(--line);
    border-radius: 8px;
    height: 42px;
    padding: 0 1rem;
  }
  .stTabs [data-baseweb="tab"] p {
    color: var(--ink);
  }
  .stTabs [aria-selected="true"] {
    background: #155e75;
    color: white;
    border-color: transparent;
  }
  .stTabs [aria-selected="true"] p {
    color: white;
  }
  .hero-shell h1,
  .hero-shell p,
  .hero-badge,
  .hero-badge * {
    color: white !important;
  }
</style>
"""

DEFAULTS = {
    'B': 10.0, 'L': 12.0, 'spessore': 0.8, 'E_cls_MPa': 30000.0,
    'k_winkler_kPa_m': 15000.0, 'mesh_size': 1.0, 'q_distribuito_kPa': 0.0,
    'stratigrafia_csv': DEFAULT_STRAT_PLATEA,
    'q_amm': 300.0,
    'kh': 0.15,
    'kv': 0.05,
}

DEFAULT_COLUMNS = pd.DataFrame([
    {"x": 2.0, "y": 2.0, "P_kN": 1200.0, "Mx_kNm": 50.0, "My_kNm": -30.0},
    {"x": 8.0, "y": 2.0, "P_kN": 1500.0, "Mx_kNm": 0.0, "My_kNm": 0.0},
    {"x": 2.0, "y": 10.0, "P_kN": 1100.0, "Mx_kNm": 0.0, "My_kNm": 0.0},
    {"x": 8.0, "y": 10.0, "P_kN": 1600.0, "Mx_kNm": -40.0, "My_kNm": 25.0},
])

st.set_page_config(page_title='PlateaFEM', layout='wide')
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <section class="hero-shell">
      <h1>PlateaFEM</h1>
      <p>Analisi di platee di fondazione su suolo alla Winkler con il Metodo degli Elementi Finiti (FEM), basato su OpenSeesPy. Include analisi statica e sismica.</p>
      <div class="hero-badges">
        <span class="hero-badge">FEM (OpenSeesPy)</span>
        <span class="hero-badge">Shell MITC4</span>
        <span class="hero-badge">Suolo alla Winkler</span>
        <span class="hero-badge">Analisi Sismica</span>
      </div>
    </section>
    """, unsafe_allow_html=True)

# Gestione stato e defaults per import/export
defaults = DEFAULTS.copy()
pilastri_defaults_df = DEFAULT_COLUMNS.copy()

with st.sidebar:
    st.header('📂 Import / Export')
    up = st.file_uploader('Carica configurazione (JSON)', type=['json'])
    if up is not None:
        try:
            loaded_data = json.load(up)
            defaults.update(loaded_data)
            if 'pilastri_df' in loaded_data and isinstance(loaded_data['pilastri_df'], list):
                pilastri_defaults_df = pd.DataFrame(loaded_data['pilastri_df'])
            if 'stratigrafia_csv' in loaded_data:
                defaults['stratigrafia_csv'] = loaded_data['stratigrafia_csv']
            st.success('Dati caricati con successo!')
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            st.error(f'Errore nel caricamento del file JSON: {e}')

    if not reporting_enabled:
        st.warning(
            "Librerie per report non trovate (es. `python-docx`, `matplotlib`). "
            "Installale per abilitare la generazione di report Word."
        )

    st.header('📐 Geometria Platea')
    B = st.number_input('Dimensione B [m]', 0.5, 100.0, float(defaults['B']), 0.5)
    L = st.number_input('Dimensione L [m]', 0.5, 100.0, float(defaults['L']), 0.5)
    spessore = st.number_input('Spessore Platea H [m]', 0.2, 5.0, float(defaults['spessore']), 0.1)
    
    st.header('⚙️ Proprietà Materiali e Terreno')
    E_cls_MPa = st.number_input('Modulo E Cls [MPa]', 10000.0, 60000.0, float(defaults['E_cls_MPa']), 1000.0)
    q_amm = st.number_input('Pressione ammissibile q_amm [kPa]', 50.0, 5000.0, float(defaults['q_amm']), 10.0)
    
    st.subheader('Stratigrafia')
    st.caption("Usata per stimare il modulo di Winkler. Righe: spessore,γ_dry,γ_sat,phi,cu,E_ed")
    stratigrafia_csv = st.text_area("Stratigrafia", value=defaults['stratigrafia_csv'], height=120)
    strat_df, strat_err = parse_stratigrafia_platea(stratigrafia_csv)
    
    k_winkler_mode = st.radio("Definizione Modulo di Winkler", ["Calcola da stratigrafia", "Manuale"], horizontal=True)
    
    if k_winkler_mode == "Calcola da stratigrafia":
        if not strat_err:
            k_winkler_stimato = stima_k_winkler_da_stratigrafia(strat_df, B, 0.2)
            st.info(f"Ks stimato: **{k_winkler_stimato:.0f} kPa/m**")
            k_winkler_kPa_m = k_winkler_stimato
        else:
            st.error("Stratigrafia non valida per calcolare Ks.")
            k_winkler_kPa_m = float(defaults['k_winkler_kPa_m'])
    else:
        k_winkler_kPa_m = st.number_input('Modulo di Winkler ks [kPa/m]', 1000.0, 500000.0, float(defaults['k_winkler_kPa_m']), 1000.0)

    st.header('FEM')
    mesh_size = st.number_input('Dimensione Mesh [m]', 0.2, 5.0, float(defaults['mesh_size']), 0.1)

    st.header('Modello di Calcolo')
    modello_calcolo = st.radio("Modello di Analisi", ["FEM (flessibile)", "Rigido (analitico)"], horizontal=True)

    st.header('🏛️ Carichi Pilastri')
    st.caption('Definire posizione e carichi dei pilastri. Le coordinate (0,0) sono nell\'angolo in basso a sinistra.')
    
    edited_columns_df = st.data_editor(
        pilastri_defaults_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "x": st.column_config.NumberColumn("x [m]", format="%.2f"),
            "y": st.column_config.NumberColumn("y [m]", format="%.2f"),
            "P_kN": st.column_config.NumberColumn("P [kN]", format="%.1f"),
            "Mx_kNm": st.column_config.NumberColumn("Mx [kNm]", format="%.1f"),
            "My_kNm": st.column_config.NumberColumn("My [kNm]", format="%.1f"),
        },
        key="columns_editor"
    )

    st.header('➕ Carichi Aggiuntivi')
    q_distribuito_kPa = st.number_input('Carico distribuito q [kPa]', -100.0, 100.0, float(defaults.get('q_distribuito_kPa', 0.0)), 1.0)

    st.header('🌍 Sismica')
    kh = st.number_input('kh [-]', 0.0, 1.0, float(defaults['kh']), 0.01)
    kv = st.number_input('kv [-]', 0.0, 1.0, float(defaults['kv']), 0.01)

    # Pulsante di download per l'export
    st.divider()
    current_input_data = {
        'B': B, 'L': L, 'spessore': spessore, 'E_cls_MPa': E_cls_MPa,
        'k_winkler_kPa_m': k_winkler_kPa_m, 'mesh_size': mesh_size, 'q_distribuito_kPa': q_distribuito_kPa, 'stratigrafia_csv': stratigrafia_csv,
        'q_amm': q_amm, 'kh': kh, 'kv': kv,
        'pilastri_df': edited_columns_df.to_dict('records')
    }
    st.download_button(
        label="Scarica configurazione (JSON)",
        data=json.dumps(current_input_data, indent=2),
        file_name="config_platea.json",
        mime="application/json"
    )

# --- Static Analysis ---
dati_stat = DatiPlatea(
    B=B, L=L, spessore=spessore, E_cls_MPa=E_cls_MPa,
    k_winkler_kPa_m=k_winkler_kPa_m, mesh_size=mesh_size,
    pilastri_df=edited_columns_df, q_distribuito_kPa=q_distribuito_kPa
)
err_stat = valida_dati_platea(dati_stat)
if err_stat:
    for e in err_stat:
        st.error(f"[Statico] {e}")
    st.stop()

# --- Seismic Analysis ---
pilastri_sismici_df = edited_columns_df.copy()
if not pilastri_sismici_df.empty:
    pilastri_sismici_df['P_kN'] *= (1 - kv)
    pilastri_sismici_df['Mx_kNm'] *= (1 + kh)
    pilastri_sismici_df['My_kNm'] *= (1 + kh)

q_distribuito_sismico = q_distribuito_kPa * (1 - kv)

dati_sis = DatiPlatea(
    B=B, L=L, spessore=spessore, E_cls_MPa=E_cls_MPa,
    k_winkler_kPa_m=k_winkler_kPa_m, mesh_size=mesh_size,
    pilastri_df=pilastri_sismici_df, q_distribuito_kPa=q_distribuito_sismico
)
err_sis = valida_dati_platea(dati_sis)
if err_sis:
    for e in err_sis:
        st.error(f"[Sismico] {e}")
    st.stop()

# --- Calculation ---
try:
    with st.spinner(f"Analisi in corso con modello {modello_calcolo}..."):
        if modello_calcolo == "FEM (flessibile)":
            risultati_stat = calcola_platea_fem(dati_stat)
            risultati_sis = calcola_platea_fem(dati_sis)
        else: # Rigido
            risultati_stat = calcola_platea_rigida(dati_stat)
            risultati_sis = calcola_platea_rigida(dati_sis)

    # --- Sintesi e risultati tabellati ---
    st.subheader("Risultati principali")
    p_max_stat = risultati_stat['pressioni_kPa'].max().max()
    p_max_sis = risultati_sis['pressioni_kPa'].max().max()
    ced_max_stat = risultati_stat['cedimenti_mm'].max().max()
    ced_max_sis = risultati_sis['cedimenti_mm'].max().max()
    sintesi_df = tabella_sintesi_platea(dati_stat, risultati_stat, risultati_sis, q_amm)
    verifiche_df = genera_verifiche_platea(dati_stat, risultati_stat, risultati_sis, q_amm)
    risultati_principali = pd.DataFrame([
        {"Parametro": "Pressione max statica", "Valore": f"{p_max_stat:.2f}", "Unita": "kPa", "Esito/nota": f"Limite {q_amm:.2f}"},
        {"Parametro": "Pressione max sismica", "Valore": f"{p_max_sis:.2f}", "Unita": "kPa", "Esito/nota": f"Limite {q_amm * 1.25:.2f}"},
        {"Parametro": "Cedimento max statico", "Valore": f"{ced_max_stat:.2f}", "Unita": "mm", "Esito/nota": "-"},
        {"Parametro": "Cedimento max sismico", "Valore": f"{ced_max_sis:.2f}", "Unita": "mm", "Esito/nota": "-"},
    ])
    st.dataframe(risultati_principali, use_container_width=True, hide_index=True)

    if p_max_stat > q_amm or p_max_sis > q_amm * 1.25:
        st.error("Verifica a pressione massima NON soddisfatta.")
    else:
        st.success("Verifica a pressione massima soddisfatta.")

    if modello_calcolo == "Rigido (analitico)":
        st.info("L'analisi con modello rigido non calcola le sollecitazioni interne (momenti). Vengono mostrate solo le pressioni e un cedimento medio.")

    sez_geometria, sez_statica, sez_sismica, sez_verifiche, sez_report = [st.container() for _ in range(5)]

    with sez_geometria:
        st.subheader("Geometria, mesh FEM e posizione carichi")
        if modello_calcolo == "FEM (flessibile)":
            st.plotly_chart(figura_geometria_platea(dati_stat, risultati_stat), use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("La visualizzazione della mesh e' disponibile solo per il modello FEM.")

    with sez_statica:
        st.subheader("Mappe dei risultati - condizione statica")
        st.plotly_chart(figura_risultati_platea(dati_stat, risultati_stat, 'cedimenti_mm', 'Cedimenti [mm]'), use_container_width=True)
        st.plotly_chart(figura_risultati_platea(dati_stat, risultati_stat, 'pressioni_kPa', 'Pressioni [kPa]'), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figura_risultati_platea(dati_stat, risultati_stat, 'Mxx_kNm_m', 'Momento Mxx [kNm/m]'), use_container_width=True)
        with col2:
            st.plotly_chart(figura_risultati_platea(dati_stat, risultati_stat, 'Myy_kNm_m', 'Momento Myy [kNm/m]'), use_container_width=True)

    with sez_sismica:
        st.subheader("Mappe dei risultati - condizione sismica")
        st.plotly_chart(figura_risultati_platea(dati_sis, risultati_sis, 'cedimenti_mm', 'Cedimenti [mm]'), use_container_width=True)
        st.plotly_chart(figura_risultati_platea(dati_sis, risultati_sis, 'pressioni_kPa', 'Pressioni [kPa]'), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figura_risultati_platea(dati_sis, risultati_sis, 'Mxx_kNm_m', 'Momento Mxx [kNm/m]'), use_container_width=True)
        with col2:
            st.plotly_chart(figura_risultati_platea(dati_sis, risultati_sis, 'Myy_kNm_m', 'Momento Myy [kNm/m]'), use_container_width=True)

    with sez_verifiche:
        st.subheader("Verifiche e sintesi tecnica")
        st.dataframe(sintesi_df, use_container_width=True, hide_index=True)
        st.dataframe(verifiche_df, use_container_width=True, hide_index=True)
        for note in genera_note_platea(modello_calcolo):
            st.info(note)

    if reporting_enabled:
        with sez_report:
            st.subheader("Relazione tecnica Word")
            try:
                report_bytes = create_word_report(dati_stat, risultati_stat, dati_sis, risultati_sis, q_amm, modello_calcolo)
                st.download_button(
                    "Scarica relazione (.docx)",
                    data=report_bytes,
                    file_name="relazione_platea.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception as report_exc:
                st.error(f"Generazione relazione non riuscita: {report_exc}")

except Exception as e:
    st.error(f"Errore critico durante l'analisi FEM: {e}")
    st.exception(e)
