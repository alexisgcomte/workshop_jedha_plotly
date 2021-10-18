import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ecg_qc import EcgQc
import math
import numpy as np
from modules.RR_detection import compute_heart_rate


def ecg_graph_generation(df: pd.DataFrame,
                         tick_space: int = 9,
                         fs: int = 1_000,
                         additional_display: str = 'consensus',
                         algo: str = 'hamilton',
                         time_window_ml=2) -> go.Figure:

    # ecg_qc predictions
    classif_ecg_qc_ml_data = ecg_qc_predict(ecg_data=df['ecg_signal'].values,
                                            time_window_ml=time_window_ml,
                                            fs=fs)

    # Creating binary class for quality classification
    for column in df.columns[1:]:
        df.loc[:, column] = df[column].apply(
            lambda x: annot_classification_correspondance(x))

    # Preparing data
    data = np.transpose(df.iloc[:, 1:5].values)
    data = [data[0],
            data[1],
            data[2],
            data[3],
            classif_ecg_qc_ml_data]

    # Plotly initialization
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Ecg Line generation
    ecg_signal_line = go.Scatter(x=df.index/fs,
                                 y=df['ecg_signal'],
                                 mode='lines',
                                 name='ecg_signal',
                                 marker_color='rgba(44, 202, 82, .8)')
    fig.add_trace(ecg_signal_line, secondary_y=False)

    # HR Computation
    compute_hr = compute_heart_rate(fs=fs)
    compute_hr.compute(df)
    hr_list = compute_hr.data[algo]['hr']
    qrs_list = compute_hr.data[algo]['qrs']/fs
    # rr_list = compute_hr.data[algo]['rr_intervals']

    # QRS detection display
    y_max = max(ecg_signal_line.y)

    for _, frame in enumerate(qrs_list):
        fig.add_shape(type='line',
                      yref="y",
                      xref="x",
                      x0=frame,
                      x1=frame,
                      y0=y_max*0.9,
                      y1=y_max,
                      line=dict(color='blue', width=2),
                      name='detection')

    if additional_display == 'consensus':
        labels = list(df.iloc[:, 1:5].columns) + ['ecg_qc pred ']
        fig.add_trace(go.Heatmap(x=df.index/fs,
                                 y=labels,
                                 z=data,
                                 colorscale=[[0.0, "rgb(160,0,0)"],
                                             [1.0, "rgb(0,140,0)"]],
                                 zmin=0,
                                 zmax=1,
                                 opacity=0.2,
                                 showscale=False,
                                 ygap=1),
                      secondary_y=True)

    elif additional_display == 'heart rate':
        fig.add_trace(go.Scatter(x=qrs_list,
                                 y=hr_list,
                                 mode='lines',
                                 name='hr_list',
                                 marker_color='rgba(44, 117, 255, .8)'),
                      secondary_y=True)

        fig.update_yaxes(range=[0, 100], secondary_y=True)

    fig.update_layout(template='plotly_white',
                      title='ECG viz',
                      xaxis_title='Seconds',
                      # yaxis2=dict(range=[0, 50]),
                      xaxis=dict(showgrid=True,
                                  tickmode='linear',
                                  ticks="inside",
                                  # ticklabelposition='inside top',
                                  tickson="boundaries",
                                  tick0=df.index[0]/fs,
                                  ticklen=10,
                                  tickwidth=1,
                                  dtick=tick_space,
                                  side='top'),
                      yaxis=dict(fixedrange=True),
                      yaxis2=dict(fixedrange=True)
                      )

    return fig


def ecg_qc_predict(ecg_data: np.ndarray,
                   time_window_ml: int = 2,
                   fs: int = 1_000) -> np.ndarray:

    ecg_qc_test = EcgQc(model='model/tuh_model_2s.joblib',
                        sampling_frequency=1000,
                        normalized=False)
    classif_ecg_qc_data = np.zeros(len(ecg_data))

    for start in range(
            math.floor(len(ecg_data)/(fs * time_window_ml)) + 1):

        start = start * fs * time_window_ml
        end = start + fs * time_window_ml

        signal_quality = ecg_qc_test.predict_quality(
            ecg_qc_test.compute_sqi_scores(ecg_data[start:end]))

        classif_ecg_qc_data[start:end] = signal_quality

    return classif_ecg_qc_data


def annot_classification_correspondance(classif: int) -> int:

    if classif in [2, 3]:
        classif_correspondance = 0
    else:
        classif_correspondance = 1

    return classif_correspondance
