import streamlit as st
from dataclasses import dataclass
from typing import Optional
import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ═════════════════════════════════════════════════════════════
# 1) CONFIGURATION DATACLASS
# ═════════════════════════════════════════════════════════════
@dataclass
class Config:
    data_file: pathlib.Path | str
    rolling_win: Optional[int]
    p_low: float
    p_high: float
    w_min: float
    w_mid: float
    w_max: float
    ann_days: int = 252

# constante anti-division par zéro
_EPS = 1e-12

# ═════════════════════════════════════════════════════════════
# 2) DATA FUNCTIONS
# ═════════════════════════════════════════════════════════════
def load_data(data_file):
    df = pd.read_excel(data_file, sheet_name='Data', skiprows=3, header=0)
    df = df[['Dates', 'SPGSPM Index', 'SPGSIN Index', 'RoRo Index']].copy()
    df.columns = ['date', 'precious', 'industrial', 'roro']
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.dropna(subset=['date','precious','industrial','roro'])\
           .sort_values('date').reset_index(drop=True)
    # rendements journaliers
    df['prec_ret'] = df['precious'].pct_change()
    df['ind_ret']  = df['industrial'].pct_change()
    return df

# ═════════════════════════════════════════════════════════════
# 3) WEIGHTS FUNCTION
# ═════════════════════════════════════════════════════════════
def capped_weights(roro, p_low, p_high, w_min, w_mid, w_max):
    mid = (p_low + p_high) / 2
    low = w_min + (w_mid - w_min) * (roro - p_low) / (mid - p_low)
    high = w_mid + (w_max - w_mid) * (roro - mid) / (p_high - mid)
    w = np.where(roro <= mid, low, high)
    return pd.Series(w, index=roro.index).clip(w_min, w_max)

# ═════════════════════════════════════════════════════════════
# 4) METRICS FUNCTION
# ═════════════════════════════════════════════════════════════
def compute_metrics(nav, ret, w_prec, dates, ann=252):
    nb_years = len(ret.dropna()) / ann
    total = nav.iloc[-1] - 1
    cagr  = nav.iloc[-1] ** (1 / nb_years) - 1
    vol   = ret.std() * np.sqrt(ann)
    sharpe   = ret.mean() / max(ret.std(), _EPS) * np.sqrt(ann)
    downside = ret.clip(upper=0).std()
    sortino  = ret.mean() / max(downside, _EPS) * np.sqrt(ann)
    mdd   = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(mdd) if mdd else np.nan

    s     = pd.Series(ret.values, index=dates)
    best  = ((1 + s).resample('YE').prod() - 1).max()
    worst = ((1 + s).resample('YE').prod() - 1).min()
    pos_mon = ((1 + s).resample('ME').prod() - 1 > 0).mean()
    turnover = w_prec.diff().abs().sum() / 2 / nb_years

    return pd.Series({
        'TotalRet %': 100 * total,
        'CAGR %'    : 100 * cagr,
        'Vol %'     : 100 * vol,
        'Sharpe'    : sharpe,
        'Sortino'   : sortino,
        'MaxDD %'   : 100 * mdd,
        'Calmar'    : calmar,
        'BestYear %': 100 * best,
        'WorstYear %':100 * worst,
        '%PosMonths':100 * pos_mon,
        'Turnover/yr %':100 * turnover
    }).round(2)

# ═════════════════════════════════════════════════════════════
# 5) STREAMLIT APP
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title='RoRo Interactive Dashboard', layout='wide')
st.title('RoRo Strategy Interactive Dashboard')

# Sidebar: upload + parameters
uploaded = st.sidebar.file_uploader('Upload Excel file', type=['xlsx'])
if uploaded:
    # Paramètres
    use_rolling = st.sidebar.checkbox('Use rolling window')
    rolling_win = st.sidebar.number_input('Rolling window (days)',
                                          min_value=1, value=756) if use_rolling else None
    p_low  = st.sidebar.slider('Percentile low (P10)',  0.0, 0.5, 0.10)
    p_high = st.sidebar.slider('Percentile high (P80)', 0.5, 1.0, 0.80)
    w_min = st.sidebar.slider('Weight min', 0.0, 1.0, 0.20)
    w_mid = st.sidebar.slider('Weight mid', 0.0, 1.0, 0.40)
    w_max = st.sidebar.slider('Weight max', 0.0, 1.0, 0.80)

    # Choix graphiques / tableaux
    graph_options = [
        'NAV cumulatif',
        'Indice RoRo',
        'Allocation dynamique',
        'Évolution des poids',
        'Distribution des poids',
        'Dernier mois poids'
    ]
    table_options = ['Performance Metrics', 'Poids par période']

    selected_graphs = st.sidebar.multiselect('Choisir graphiques', graph_options)
    selected_tables = st.sidebar.multiselect('Choisir tableaux', table_options)

    # Chargement
    df = load_data(uploaded)

    # Percentiles fixes ou roulantes
    if use_rolling:
        roll = df['roro'].rolling(rolling_win, min_periods=rolling_win)
        p_low_full  = roll.quantile(p_low)
        p_high_full = roll.quantile(p_high)
        p_low_val   = p_low_full.iloc[-1]
        p_high_val  = p_high_full.iloc[-1]
    else:
        p_low_val, p_high_val = df['roro'].quantile([p_low, p_high])
        p_low_full, p_high_full = p_low_val, p_high_val

    # Poids & NAV
    df['w_prec'] = capped_weights(df['roro'], p_low_full, p_high_full, w_min, w_mid, w_max)
    df['w_ind']  = 1 - df['w_prec']
    df = df.dropna(subset=['w_prec'])
    df['strat_ret'] = df['w_prec'] * df['prec_ret'] + df['w_ind'] * df['ind_ret']
    df['ws_ret']    = 0.40 * df['prec_ret'] + 0.60 * df['ind_ret']
    for col in ['strat_ret','ws_ret','prec_ret','ind_ret']:
        df[f'{col}_nav'] = (1 + df[col]).cumprod()

    cfg = Config(uploaded, rolling_win, p_low, p_high, w_min, w_mid, w_max)

    # ── TABLEAUX ────────────────────────────────────────────────────
    if 'Performance Metrics' in selected_tables:
        metrics = pd.DataFrame({
            'RoRo (20-40-80)': compute_metrics(df['strat_ret_nav'], df['strat_ret'], df['w_prec'], df['date']),
            'Statique 40/60' : compute_metrics(df['ws_ret_nav'],   df['ws_ret'],  pd.Series(0.40,index=df.index), df['date']),
            'Précieux 100%'  : compute_metrics(df['prec_ret_nav'], df['prec_ret'], pd.Series(1.0, index=df.index), df['date']),
            'Industriel 100%': compute_metrics(df['ind_ret_nav'],  df['ind_ret'],  pd.Series(0.0, index=df.index), df['date'])
        }).T
        st.subheader('Performance Metrics')
        st.dataframe(metrics)

    if 'Poids par période' in selected_tables:
        start, end = st.sidebar.date_input('Période Poids',
                                           [df['date'].min(), df['date'].max()])
        mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
        table = df.loc[mask, ['date','roro','w_prec','w_ind']].copy()
        # afficher les plus récentes en premier
        table = table.sort_values('date', ascending=False)
        table['Date']   = table['date'].dt.strftime('%Y-%m-%d')
        table['RoRo']   = table['roro'].map('{:.2f}'.format)
        table['Prec %'] = (table['w_prec'] * 100).round(2).astype(str) + '%'
        table['Ind %']  = (table['w_ind']  * 100).round(2).astype(str) + '%'
        display = table[['Date','RoRo','Prec %','Ind %']]
        st.subheader(f'Poids du {start} au {end}')
        st.dataframe(display)

    # ── GRAPHIQUES ──────────────────────────────────────────────────
    for g in selected_graphs:

        if g == 'NAV cumulatif':
            scale = st.sidebar.radio('Échelle NAV', ['Linéaire','Log'], index=1)
            fig = px.line(df, x='date',
                          y=['strat_ret_nav','ws_ret_nav','prec_ret_nav','ind_ret_nav'],
                          labels={'value':'NAV','date':'Date','variable':'Stratégie'},
                          title='NAV cumulatif')
            if scale == 'Log':
                fig.update_yaxes(type='log')
            fig.update_layout(legend_title_text='Stratégie')
            st.plotly_chart(fig, use_container_width=True)

        if g == 'Indice RoRo':
            # choix de la plage
            period = st.sidebar.radio(
                'Période Indice RoRo',
                ['Période complète', 'Dernier mois'],
                key='roro_period'
            )
            if period == 'Dernier mois':
                data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
            else:
                data = df

            median = data['roro'].median()
            fig = go.Figure()

            # ligne de médiane
            fig.add_trace(go.Scatter(
                x=data['date'], y=[median]*len(data),
                mode='lines', name=f'Médiane {median:.2f}',
                line=dict(dash='dash', color='grey', width=1)
            ))

            # zones colorées (sous / au-dessus de la médiane)
            below = np.where(data['roro'] < median, data['roro'], median)
            fig.add_trace(go.Scatter(
                x=data['date'], y=below,
                mode='lines', name='Below Median',
                line=dict(width=0),
                fill='tonexty', fillcolor='rgba(0,126,51,0.3)'
            ))
            above = np.where(data['roro'] > median, data['roro'], median)
            fig.add_trace(go.Scatter(
                x=data['date'], y=above,
                mode='lines', name='Above Median',
                line=dict(width=0),
                fill='tonexty', fillcolor='rgba(213,0,0,0.3)'
            ))

            # courbe RoRo légère
            fig.add_trace(go.Scatter(
                x=data['date'], y=data['roro'],
                mode='lines', name='RoRo',
                line=dict(color='rgba(0,0,0,0.5)', width=2)
            ))

            # percentiles roulants ou fixes
            if isinstance(p_low_full, pd.Series):
                # aligne les percentiles sur la période sélectionnée
                p10 = p_low_full.loc[data.index]
                p80 = p_high_full.loc[data.index]
                fig.add_trace(go.Scatter(
                    x=data['date'], y=p10,
                    mode='lines', name=f'P{int(p_low*100)} roul.',
                    line=dict(dash='dot', color='firebrick', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data['date'], y=p80,
                    mode='lines', name=f'P{int(p_high*100)} roul.',
                    line=dict(dash='dot', color='darkgreen', width=1)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=data['date'], y=[p_low_val]*len(data),
                    mode='lines', name=f'P{int(p_low*100)} ({p_low_val:.2f})',
                    line=dict(dash='dot', color='firebrick', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data['date'], y=[p_high_val]*len(data),
                    mode='lines', name=f'P{int(p_high*100)} ({p_high_val:.2f})',
                    line=dict(dash='dot', color='darkgreen', width=1)
                ))

            fig.update_layout(
                title='Indice RoRo',
                xaxis_title='Date',
                yaxis_title='RoRo',
                legend_title_text='',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)



        if g == 'Allocation dynamique':
            fig = go.Figure()
            # Allocation stacked
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['w_prec'],
                mode='lines', fill='tozeroy', name='Precious'
            ))
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['w_ind'],
                mode='lines', fill='tonexty', name='Industrial'
            ))
            # Overlay RoRo index
            fig.add_trace(go.Scatter(
                x=df['date'], y=df['roro'],
                mode='lines', name='RoRo Index',
                line=dict(color='black', dash='dash'),
                yaxis='y2'
            ))
            fig.update_layout(
                title='Allocation Dynamique & Indice RoRo',
                xaxis_title='Date',
                yaxis_title='Poids',
                legend_title_text='',
                yaxis2=dict(
                    title='RoRo',
                    overlaying='y',
                    side='right'
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        if g == 'Évolution des poids':
            fig = px.line(df, x='date',
                          y=[df['w_prec']*100, df['w_ind']*100],
                          labels={'value':'Poids (%)','date':'Date','variable':'Asset'},
                          title='Évolution des poids (%)')
            fig.update_layout(legend_title_text='Asset')
            st.plotly_chart(fig, use_container_width=True)

        if g == 'Distribution des poids':
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['w_prec']*100, name='Precious', opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df['w_ind']*100,  name='Industrial', opacity=0.7
            ))
            fig.update_layout(
                barmode='overlay',
                title='Distribution des poids (%)',
                legend_title_text='Asset'
            )
            st.plotly_chart(fig, use_container_width=True)

        if g == 'Dernier mois poids':
            last = df[df['date'] >= df['date'].max() - pd.Timedelta(days=30)]
            fig = px.line(last, x='date',
                          y=[last['w_prec']*100, last['w_ind']*100],
                          labels={'value':'Poids (%)','date':'Date','variable':'Asset'},
                          title='Poids – Dernier mois')
            fig.update_layout(legend_title_text='Asset')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info('Veuillez uploader un fichier Excel pour démarrer.')
