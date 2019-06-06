#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 19:46:00 2019

@author: minghao
"""

import plotly.graph_objs as go
import plotly.offline as py

import numpy as np

data = [go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color = np.random.randn(500),
        colorscale='Viridis',
        showscale=True,
 line={'width': 5}
    )
)]

layout = dict(
plot_bgcolor ='#F5F7FA',
   paper_bgcolor = '#F5F7FA',
   width = 500,
xaxis = {'zeroline': False},
yaxis = {'zeroline': False},
)

py.plot( dict(data=data, layout=layout) )