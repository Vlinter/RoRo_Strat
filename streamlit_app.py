import streamlit as st
from dataclasses import dataclass
from typing import Optional
import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# 1) DATACLASS & CONSTANTE
# ──────────────────────────────────────────────────────────────────────────────
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

_EPS = 1e-12  # anti-division by zero

# ──────────────────────────────────────────────────────────────────────────────
# 2) FONCTIONS DE PRÉPARATION
# ──────────────────────────────────────────────────────────────────────────────
def load_data(data_file):
    df = (
        pd.read_excel(data_file, sheet_name='Data', skiprows=3)
          [['Dates','SPGSPM Index','SPGSIN Index','RoRo Index']]
          .rename(columns={
              'Dates':'date',
              'SPGSPM Index':'precious',
              'SPGSIN Index':'industrial',
              'RoRo Index':'roro'
          })
    )
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.dropna(subset=['date','precious','industrial','roro']) \
           .sort_values('date').reset_index(drop=True)
    df['prec_ret'] = df['precious'].pct_change()
    df['ind_ret']  = df['industrial'].pct_change()
    return df

def capped_weights(roro, p_low, p_high, w_min, w_mid, w_max):
    mid  = (p_low + p_high)/2
    low  = w_min + (w_mid-w_min)*(roro-p_low)/(mid-p_low)
    high = w_mid + (w_max-w_mid)*(roro-mid)/(p_high-mid)
    w = np.where(roro<=mid, low, high)
    return pd.Series(w, index=roro.index).clip(w_min, w_max)

def compute_metrics(nav, ret, w_prec, dates, ann=252):
    nyr     = len(ret.dropna())/ann
    total   = nav.iloc[-1] - 1
    cagr    = nav.iloc[-1]**(1/nyr) - 1
    vol     = ret.std() * np.sqrt(ann)
    sharpe  = ret.mean() / max(ret.std(),_EPS) * np.sqrt(ann)
    downside= ret.clip(upper=0).std()
    sortino = ret.mean() / max(downside,_EPS) * np.sqrt(ann)
    mdd     = (nav/nav.cummax() - 1).min()
    calmar  = cagr/abs(mdd) if mdd else np.nan

    s      = pd.Series(ret.values, index=dates)
    best   = ((1+s).resample('YE').prod() - 1).max()
    worst  = ((1+s).resample('YE').prod() - 1).min()
    posm   = ((1+s).resample('ME').prod() - 1 > 0).mean()
    turnover = w_prec.diff().abs().sum()/2/nyr

    return pd.Series({
        'TotalRet %'    :100*total,
        'CAGR %'        :100*cagr,
        'Vol %'         :100*vol,
        'Sharpe'        :sharpe,
        'Sortino'       :sortino,
        'MaxDD %'       :100*mdd,
        'Calmar'        :calmar,
        'BestYear %'    :100*best,
        'WorstYear %'   :100*worst,
        '%PosMonths'    :100*posm,
        'Turnover/yr %':100*turnover
    }).round(2)

# ──────────────────────────────────────────────────────────────────────────────
# 3) STREAMLIT APP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='RoRo Dashboard', layout='wide')
st.title('RoRo Interactive Dashboard')

uploaded = st.sidebar.file_uploader('Upload Excel file', type=['xlsx'])
if not uploaded:
    st.info('Veuillez uploader un fichier Excel pour démarrer.')
    st.stop()

# Paramètres
use_roll    = st.sidebar.checkbox('Use rolling window')
rolling_win = st.sidebar.number_input('Rolling window (jours)', 1, 2000, 756) if use_roll else None
p_low       = st.sidebar.slider('P bas (P10)',  0.0, 0.5, 0.10)
p_high      = st.sidebar.slider('P haut (P80)', 0.5, 1.0, 0.80)
w_min       = st.sidebar.slider('Poids min', 0.0, 1.0, 0.25)
w_mid       = st.sidebar.slider('Poids mid', 0.0, 1.0, 0.40)
w_max       = st.sidebar.slider('Poids max', 0.0, 1.0, 0.60)

# Options de graphiques et tableaux
graph_opts = [
    'NAV cumulatif',
    'Indice RoRo',
    'Allocation dynamique',
    'Évolution des indices',
    'Distribution des poids'
]
table_opts = ['Performance Metrics', 'Poids par période']

sel_graphs = st.sidebar.multiselect('Graphiques', graph_opts)
sel_tables = st.sidebar.multiselect('Tableaux', table_opts)

# Chargement & prépa
df = load_data(uploaded)
if use_roll:
    roll    = df['roro'].rolling(rolling_win, min_periods=rolling_win)
    p_low_f = roll.quantile(p_low)
    p_high_f= roll.quantile(p_high)
    p_low_v = p_low_f.iloc[-1]
    p_high_v= p_high_f.iloc[-1]
else:
    p_low_v, p_high_v = df['roro'].quantile([p_low,p_high])
    p_low_f, p_high_f = p_low_v, p_high_v

df['w_prec'] = capped_weights(df['roro'], p_low_f, p_high_f, w_min, w_mid, w_max)
df['w_ind']  = 1 - df['w_prec']
df = df.dropna(subset=['w_prec'])
df['strat_ret'] = df['w_prec']*df['prec_ret'] + df['w_ind']*df['ind_ret']
df['ws_ret']    = 0.40*df['prec_ret'] + 0.60*df['ind_ret']
for c in ['strat_ret','ws_ret','prec_ret','ind_ret']:
    df[f'{c}_nav'] = (1+df[c]).cumprod()


# ── OPTIMISATION PERCENTILES ────────────────────────────────────────────────
optim = st.sidebar.checkbox("🔍 Optimisation percentiles fixes")
if optim:
    st.subheader("Optimisation des percentiles fixes")

    # 1) Définition des inputs manquants
    cost_sw = st.sidebar.number_input(
        "Coût par switch (fraction, ex. 0.0025 = 25 bp)",
        min_value=0.0, max_value=0.01, value=0.0, step=0.0005,
        key="opt_cost"
    )
    crit = st.sidebar.selectbox(
        "Critère à maximiser",
        ("Sharpe", "CAGR %", "Calmar", "Sortino"),
        index=0,
        key="opt_crit"
    )
    step_q = st.sidebar.slider(
        "Pas quantile (%)",
        min_value=1, max_value=10, value=5,
        key="opt_step"
    ) / 100.0

    plows = np.arange(0.01, 0.51, step_q)
    phighs = np.arange(0.51, 1.00, step_q)

    rows = []
    for ql in plows:
        for qh in phighs:
            if ql >= qh:
                continue

            pL = df['roro'].quantile(ql)
            pH = df['roro'].quantile(qh)

            # on utilise capped_weights pour respecter w_min, w_mid, w_max
            w_prec = capped_weights(df['roro'], pL, pH, w_min, w_mid, w_max)
            w_ind  = 1 - w_prec

            ret      = w_prec * df['prec_ret'] + w_ind * df['ind_ret']
            turnover = w_prec.diff().abs() / 2
            ret      = ret - cost_sw * turnover.fillna(0)
            nav      = (1 + ret).cumprod()

            mets = compute_metrics(nav, ret, w_prec, df['date'])

            rows.append({
                "P_low":    round(ql,3),
                "P_high":   round(qh,3),
                "Sharpe":   mets["Sharpe"],
                "CAGR %":   mets["CAGR %"],
                "Calmar":   mets["Calmar"],
                "Sortino":  mets["Sortino"],
                "Vol %":    mets["Vol %"]
            })

    grid = pd.DataFrame(rows)

    # Affichage complet trié
    st.subheader(f"Toutes les combinaisons triées par **{crit}**")
    st.dataframe(grid.sort_values(crit, ascending=False).reset_index(drop=True))

    # Heatmap moyenne
    pivot = grid.pivot(index="P_low", columns="P_high", values=crit)
    fig = px.imshow(
        pivot.values,
        x=pivot.columns * 100,
        y=pivot.index * 100,
        labels={"x": "P_high (%)", "y": "P_low (%)", "color": crit},
        origin="lower",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title=f"Heatmap moyenne de {crit}"
    )
    best = grid.loc[grid[crit].idxmax()]
    fig.add_scatter(
        x=[best.P_high * 100],
        y=[best.P_low  * 100],
        mode="markers",
        marker_symbol="star",
        marker_color="cyan",
        marker_size=15,
        name="Meilleur combo"
    )
    st.plotly_chart(fig, use_container_width=True)






# ── TABLEAUX ────────────────────────────────────────────────────────────────
if 'Performance Metrics' in sel_tables:
    mets = pd.DataFrame({
        'RoRo (20-40-80)': compute_metrics(df['strat_ret_nav'], df['strat_ret'], df['w_prec'], df['date']),
        'Statique 40/60' : compute_metrics(df['ws_ret_nav'],    df['ws_ret'],  pd.Series(0.40,index=df.index), df['date']),
        'Précieux 100%'  : compute_metrics(df['prec_ret_nav'],  df['prec_ret'],pd.Series(1.0, index=df.index), df['date']),
        'Industriel 100%': compute_metrics(df['ind_ret_nav'],   df['ind_ret'], pd.Series(0.0, index=df.index), df['date'])
    }).T.drop(columns=['Turnover/yr %'])
    st.subheader('Performance Metrics')
    st.dataframe(mets)

if 'Poids par période' in sel_tables:
    start_d, end_d = st.sidebar.date_input('Période Poids', [df['date'].min().date(), df['date'].max().date()])
    start_ts, end_ts = pd.to_datetime(start_d), pd.to_datetime(end_d)
    mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
    tbl  = df.loc[mask, ['date','roro','w_prec','w_ind']]\
              .sort_values('date', ascending=False)
    tbl['Date']   = tbl['date'].dt.strftime('%Y-%m-%d')
    tbl['RoRo']   = tbl['roro'].map('{:.2f}'.format)
    tbl['Prec %'] = (tbl['w_prec']*100).round(2).astype(str)+'%'
    tbl['Ind %']  = (tbl['w_ind']*100).round(2).astype(str)+'%'
    st.subheader(f'Poids du {start_ts.date()} au {end_ts.date()}')
    st.dataframe(tbl[['Date','RoRo','Prec %','Ind %']])

# ── GRAPHIQUES ───────────────────────────────────────────────────────────────
for g in sel_graphs:

    if g == 'NAV cumulatif':
        scale = st.sidebar.radio('Échelle NAV',['Linéaire','Log'],index=1)
        fig = px.line(df, x='date',
                      y=['strat_ret_nav','ws_ret_nav','prec_ret_nav','ind_ret_nav'],
                      labels={'value':'NAV','date':'Date','variable':'Stratégie'},
                      title='NAV cumulatif')
        if scale=='Log': fig.update_yaxes(type='log')
        fig.update_layout(legend_title_text='Stratégie', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    if g == 'Indice RoRo':
        period = st.sidebar.radio('Période RoRo',['Complète','Dernier mois'],key='pr')
        data   = df[df['date']>=df['date'].max()-pd.Timedelta(days=30)] if period=='Dernier mois' else df
        med    = data['roro'].median()
        fig = go.Figure()
        # médiane
        fig.add_trace(go.Scatter(
            x=data['date'], y=[med]*len(data),
            mode='lines', name=f'Médiane {med:.2f}',
            line=dict(dash='dash',color='grey',width=1)
        ))
        # zones colorées
        below = np.where(data['roro']<med, data['roro'], med)
        above = np.where(data['roro']>med, data['roro'], med)
        fig.add_trace(go.Scatter(x=data['date'], y=below,
            mode='lines', name='Below Median',
            line=dict(width=0), fill='tonexty', fillcolor='rgba(0,126,51,0.3)'
        ))
        fig.add_trace(go.Scatter(x=data['date'], y=above,
            mode='lines', name='Above Median',
            line=dict(width=0), fill='tonexty', fillcolor='rgba(213,0,0,0.3)'
        ))
        # courbe RoRo
        fig.add_trace(go.Scatter(x=data['date'], y=data['roro'],
            mode='lines', name='RoRo',
            line=dict(color='rgba(0,0,0,0.5)',width=2)
        ))
        # percentiles roulants/fixes
        if isinstance(p_low_f, pd.Series):
            p10 = p_low_f.loc[data.index]
            p80 = p_high_f.loc[data.index]
            fig.add_trace(go.Scatter(x=data['date'], y=p10,
                mode='lines', name=f'P{int(p_low*100)} roul.',
                line=dict(dash='dot',color='firebrick',width=1)
            ))
            fig.add_trace(go.Scatter(x=data['date'], y=p80,
                mode='lines', name=f'P{int(p_high*100)} roul.',
                line=dict(dash='dot',color='darkgreen',width=1)
            ))
        else:
            fig.add_trace(go.Scatter(x=data['date'], y=[p_low_v]*len(data),
                mode='lines', name=f'P{int(p_low*100)} ({p_low_v:.2f})',
                line=dict(dash='dot',color='firebrick',width=1)
            ))
            fig.add_trace(go.Scatter(x=data['date'], y=[p_high_v]*len(data),
                mode='lines', name=f'P{int(p_high*100)} ({p_high_v:.2f})',
                line=dict(dash='dot',color='darkgreen',width=1)
            ))
        fig.update_layout(title='Indice RoRo',
                          xaxis_title='Date', yaxis_title='RoRo',
                          legend_title_text='', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    if g == 'Allocation dynamique':
        period = st.sidebar.radio('Période Allocation',['Complète','Dernier mois'],key='pa')
        data = df[df['date']>=df['date'].max()-pd.Timedelta(days=30)] if period=='Dernier mois' else df
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['date'], y=data['w_prec'],
            mode='lines', fill='tozeroy', name='Precious'
        ))
        fig.add_trace(go.Scatter(x=data['date'], y=data['w_ind'],
            mode='lines', fill='tonexty', name='Industrial'
        ))
        fig.add_trace(go.Scatter(x=data['date'], y=data['roro'],
            mode='lines', name='RoRo Index',
            line=dict(color='black',dash='dash'), yaxis='y2'
        ))
        fig.update_layout(
            title='Allocation Dynamique & Indice RoRo',
            xaxis_title='Date', yaxis_title='Poids',
            yaxis2=dict(title='RoRo',overlaying='y',side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    if g == 'Évolution des indices':
        # 1) Sélection de la période à afficher
        graph_range = st.sidebar.date_input(
            "Période Évolution indices",
            [df['date'].min().date(), df['date'].max().date()],
            key='idx_period'
        )
        start_gr = pd.to_datetime(graph_range[0])
        end_gr   = pd.to_datetime(graph_range[1])
        df_graph = df[(df['date'] >= start_gr) & (df['date'] <= end_gr)].copy()

        # 2) On normalise en Base 100 à partir de la date de début
        baseline = df_graph.iloc[0]
        base_prec = baseline['precious']
        base_ind  = baseline['industrial']

        precious_norm   = df_graph['precious']   / base_prec * 100
        industrial_norm = df_graph['industrial'] / base_ind  * 100
        ratio           = df_graph['precious']   / df_graph['industrial']

        # 3) Choix de l’échelle
        scale_type = st.sidebar.radio(
            "Échelle Indices", ["Linéaire", "Log"], index=0, key='idx_scale'
        )

        # 4) Construction du graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_graph['date'], y=precious_norm,
            mode='lines', name='Precious (Base 100)',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_graph['date'], y=industrial_norm,
            mode='lines', name='Industrial (Base 100)',
            line=dict(color='lightblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_graph['date'], y=ratio,
            mode='lines', name='Ratio Precious/Industrial',
            line=dict(color='darkorange', dash='dash', width=2),
            yaxis='y2'
        ))

        # 5) Mise en forme
        layout = dict(
            title="Évolution des indices (Base 100) & Ratio Precious/Industrial",
            xaxis=dict(title='Date'),
            yaxis=dict(title='Index (Base 100)'),
            yaxis2=dict(title='Ratio', overlaying='y', side='right'),
            legend_title_text='',
            hovermode='x unified'
        )
        if scale_type == "Log":
            layout['yaxis']['type'] = 'log'

        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)



    if g == 'Distribution des poids':
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['w_prec']*100, name='Precious', opacity=0.7))
        fig.add_trace(go.Histogram(x=df['w_ind']*100,  name='Industrial', opacity=0.7))
        fig.update_layout(
            barmode='overlay',
            title='Distribution des poids (%)',
            legend_title_text='Asset',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        
