import pandas as pd
import streamlit as st
from joblib import load
from modules.graph_generation import ecg_graph_generation


FS = 1_000
st.set_page_config(page_title="Workshop Jedha",
                   page_icon=":chart_with_upwards_trend:",
                   layout='wide',
                   initial_sidebar_state='auto')


@st.cache()
def load_ecg():

    df_ecg = pd.read_pickle('data/df_ecg_103001_selection.pkl')

    return df_ecg


def start_frame_definition():
    df_start_frame = load('streamlit/next.pkl')
    start_frame = df_start_frame.iloc[0][0]
    return df_start_frame, start_frame


# Loading in cache annotations
df_ecg = load_ecg()


# Subheader

st.sidebar.header(body='Parameters')
st.sidebar.subheader(body='Frame selection')

frame_window_selection = st.sidebar.slider(
    label='Seconds to display:',
    min_value=0,
    max_value=180,
    step=2,
    value=16)

df_start_frame, start_frame = start_frame_definition()

frame_start_selection = st.sidebar.slider(
    label='Start Frame:',
    min_value=int(round(df_ecg.index.values[0]/FS, 0)),
    max_value=int(round(df_ecg.index.values[-1]/FS, 0)),
    step=1,
    value=int(round(start_frame/FS, 0)))


start_frame = frame_start_selection * FS

tick_space_selection = st.sidebar.slider(
    label='Tick spacing (seconds):',
    min_value=1,
    max_value=10,
    step=1,
    value=2)


if st.sidebar.button('next'):
    start_frame += frame_window_selection * FS
    df_start_frame.iloc[0] = start_frame
if st.sidebar.button('previous'):
    start_frame -= frame_window_selection * FS
    df_start_frame.iloc[0] = start_frame
df_start_frame.to_pickle('streamlit/next.pkl')

end_frame = start_frame + frame_window_selection * FS
df_ecg = df_ecg[(df_ecg.index >= start_frame) & (df_ecg.index < end_frame)]

st_additional_display = st.sidebar.selectbox(
    label='additional display',
    options=['consensus', 'heart rate'])

st_algo = st.sidebar.selectbox(
    label='algo for hr detection',
    options=['xqrs', 'hamilton'])


# Graph generation
ecg_graph = st.empty()
plotly_graph = ecg_graph_generation(df=df_ecg,
                                    tick_space=tick_space_selection,
                                    fs=FS,
                                    additional_display=st_additional_display,
                                    algo=st_algo)
ecg_graph.plotly_chart(plotly_graph,
                       use_container_width=True)
