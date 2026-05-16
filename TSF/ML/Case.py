import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


aocado = pd.read_csv('../../datasets/avocado.csv')
aocado = aocado.sort_values('Date')

aocado_ts = aocado[['Date', 'AveragePrice']].rename(columns={'Date': 'ds', 'AveragePrice': 'y'})
# ensure `ds` is datetime
aocado_ts['ds'] = pd.to_datetime(aocado_ts['ds'])



try:
	import plotly
except Exception:
	print('Plotly not installed; install with: python -m pip install plotly')

from prophet import Prophet
m=Prophet()
m.fit(aocado_ts)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
plt.show()
