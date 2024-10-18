#%% Templogs / Transiciones de Fase / Calores especificos 
'''
Analizo templogs
'''
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from glob import glob
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
from datetime import datetime,timedelta
import matplotlib as mpl

#%%
def lector_templog(directorio):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura y plotea el log completo 
    '''
    data = pd.read_csv(directorio,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
        
    temp_CH1 = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2= pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp=np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 
    return timestamp,temp_CH1, temp_CH2
    
#%% FF
directorio_FF = os.path.join(os.getcwd(),'C1','241010_154839') 
paths_m_FF = glob(os.path.join(directorio_FF, '*.txt'))
paths_m_FF.sort()
paths_T_FF = glob(os.path.join(directorio_FF, '*templog*'))
# levanto fecha de archivos grabada en meta 
Fechas_FF = []

for fp in paths_m_FF:
    with open(fp, 'r') as f:
        fecha_in_file = f.readline()
        Fechas_FF.append(fecha_in_file.split()[-1])

timestamp_FF,temperatura_FF,__ = lector_templog(paths_T_FF[0])

t_full_FF = np.array([(t-timestamp_FF[0]).total_seconds() for t in timestamp_FF])
T_full_FF= temperatura_FF
dates_FF = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas_FF[:-1]] #datetimes c/ fecha de archivos 
time_delta_FF=[t.total_seconds() for t in np.diff(dates_FF)] #dif de tiempo entre archivos c resolucion 0.01 s
time_delta_FF.insert(0,0)
delta_0_FF = (dates_FF[0] - timestamp_FF[0]).total_seconds() # entre comienzo del templog y 1er archivo redondeado a .2f

#busco el indice en el templog que corresponde al segundo del 1er y ultimo dato para extrapolar tiempo y Temperatura 
indx_1er_dato_FF=np.nonzero(timestamp_FF==dates_FF[0].replace(microsecond=0))[0][0]
indx_ultimo_dato_FF=np.nonzero(timestamp_FF==datetime.strptime(Fechas_FF[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]
#Interpolo t entre tiempo de 1er y ultimo ciclo 
interp_func = interp1d(t_full_FF, T_full_FF, kind='linear')
t_interp_FF = np.round(np.arange(t_full_FF[indx_1er_dato_FF], t_full_FF[indx_ultimo_dato_FF]+1.01,0.01),2)
T_interp_FF= np.round(interp_func(t_interp_FF),2)

t_FF = np.round(delta_0_FF + np.cumsum(time_delta_FF),2)
T_FF = np.array([T_interp_FF[np.flatnonzero(t_interp_FF==t)[0]] for t in t_FF])

cmap = mpl.colormaps['jet'] #'viridis'
norm_T_FF = (np.array(T_FF) - np.array(T_FF).min()) / (np.array(T_FF).max() - np.array(T_FF).min())
colors_FF = cmap(norm_T_FF)

fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
ax.plot(t_full_FF,T_full_FF,'.-',label='Templog (Rugged O201)')
ax.plot(t_interp_FF,T_interp_FF,'-',label='Temperatura interpolada')
ax.scatter(t_FF,T_FF,color=colors_FF,label='Temperatura muestra')

plt.xlabel('t (s)')
plt.ylabel('T (°C)')
plt.legend(loc='lower right')
plt.grid()
plt.title('Temperatura de la muestra',fontsize=18)
#plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_templog.png'),dpi=300,facecolor='w')
plt.show()
#%% SV
directorio_SV = os.path.join(os.getcwd(),'SV','241010_151832') 
paths_m_SV = glob(os.path.join(directorio_SV, '*.txt'))
paths_m_SV.sort()
paths_T_SV = glob(os.path.join(directorio_SV, '*templog*'))
Fechas_SV=[]
for fp in paths_m_SV:
    with open(fp, 'r') as f:
        fecha_in_file = f.readline()
        Fechas_SV.append(fecha_in_file.split()[-1])
timestamp_SV,temperatura_SV,__ = lector_templog(paths_T_SV[0])
t_full_SV = np.array([(t-timestamp_SV[0]).total_seconds() for t in timestamp_SV])
T_full_SV= temperatura_SV
dates_SV = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas_SV[:-1]] #datetimes c/ fecha de archivos 
time_delta=[t.total_seconds() for t in np.diff(dates_SV)] #dif de tiempo entre archivos c resolucion 0.01 s
time_delta.insert(0,0)
delta_0 = (dates_SV[0] - timestamp_SV[0]).total_seconds() # entre comienzo del templog y 1er archivo redondeado a .2f
#busco el indice en el templog que corresponde al segundo del 1er y ultimo dato para extrapolar tiempo y Temperatura 
indx_1er_dato=np.nonzero(timestamp_SV==dates_SV[0].replace(microsecond=0))[0][0]
indx_ultimo_dato=np.nonzero(timestamp_SV==datetime.strptime(Fechas_SV[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]
#Interpolo t entre tiempo de 1er y ultimo ciclo 
interp_func = interp1d(t_full_SV, T_full_SV, kind='linear')
t_interp_SV = np.round(np.arange(t_full_SV[indx_1er_dato], t_full_SV[indx_ultimo_dato]+1.01,0.01),2)
T_interp_SV= np.round(interp_func(t_interp_SV),2)

time_SV = np.round(delta_0 + np.cumsum(time_delta),2)
T_SV = np.array([T_interp_SV[np.flatnonzero(t_interp_SV==t)[0]] for t in time_SV])

cmap = mpl.colormaps['jet'] #'viridis'
norm_T_SV = (np.array(T_SV) - np.array(T_SV).min()) / (np.array(T_SV).max() - np.array(T_SV).min())
colors_SV = cmap(norm_T_SV)

fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
ax.plot(t_full_SV,T_full_SV,'.-',label='Templog (Rugged O201)')
ax.plot(t_interp_SV,T_interp_SV,'-',label='Temperatura interpolada')
ax.scatter(time_SV,T_SV,color=colors_SV,label='Temperatura muestra')

plt.xlabel('t (s)')
plt.ylabel('T (°C)')
plt.legend(loc='lower right')
plt.grid()
plt.title('Temperatura de la muestra',fontsize=18)
plt.show()































#%%

fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)

ax.plot(t_interp_FF-t_interp_FF[0],T_interp_FF,'-',label='T interp FF')
ax.scatter(t_FF-t_interp_FF[0],T_FF,color=colors_FF,label='T FF')

ax.plot(t_interp_SV-t_interp_SV[0],T_interp_SV,'-',label='T interp SV')
ax.scatter(time_SV-t_interp_SV[0],T_SV,color=colors_SV,label='T FF')


plt.xlabel('t (s)')
plt.ylabel('T (°C)')
plt.legend(loc='lower right',ncol=2)
plt.grid()
plt.title('Temperatura de la muestra',fontsize=18)
plt.show()

# %%
