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
from scipy.interpolate import CubicSpline,PchipInterpolator
#%%
def lector_templog(directorio,rango_T_fijo=True):
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

def procesar_temperatura(directorio,rango_T_fijo=True):
    # Obtener archivos de datos y archivos de templog
    paths_m = glob(os.path.join(directorio, '*.txt'))
    paths_m.sort()
    paths_T = glob(os.path.join(directorio, '*templog*'))
    
    # Levantar fechas de archivos grabadas en meta
    Fechas = []
    for fp in paths_m:
        with open(fp, 'r') as f:
            fecha_in_file = f.readline()
            Fechas.append(fecha_in_file.split()[-1])
    
    # Obtener timestamps y temperaturas del templog
    timestamp, temperatura, __ = lector_templog(paths_T[0])
    
    # Calcular tiempos completos en segundos
    t_full = np.array([(t - timestamp[0]).total_seconds() for t in timestamp])
    T_full = temperatura

    # Procesar las fechas y tiempos de los archivos
    dates = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas[:-1]]  # datetimes con fecha de archivos
    time_delta = [t.total_seconds() for t in np.diff(dates)]  # diferencia de tiempo entre archivos
    time_delta.insert(0, 0)  # Insertar el primer delta como 0
    delta_0 = (dates[0] - timestamp[0]).total_seconds()  # diferencia entre comienzo templog y 1er archivo

    # Buscar los índices de los datos de templog correspondientes al primer y último archivo
    indx_1er_dato = np.nonzero(timestamp == dates[0].replace(microsecond=0))[0][0]
    indx_ultimo_dato = np.nonzero(timestamp == datetime.strptime(Fechas[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]
    
    # Interpolación entre el primer y último ciclo (a partir del 30 de Oct 24 uso PchipInterpolator)
    # interp_func = interp1d(t_full, T_full, kind='linear')
    # t_interp = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
    # T_interp = np.round(interp_func(t_interp), 2)

    # interp_func_2 = CubicSpline(t_full, T_full)
    # t_interp_2 = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
    # T_interp_2 = np.round(interp_func_2(t_interp_2), 2)

    interp_func = PchipInterpolator(t_full, T_full)
    t_interp = np.round(np.arange(t_full[indx_1er_dato], t_full[indx_ultimo_dato] + 1.01, 0.01), 2)
    T_interp = np.round(interp_func(t_interp), 2)

    # Calcular t y T a partir de los datos
    t = np.round(delta_0 + np.cumsum(time_delta), 2)
    T = np.array([T_interp[np.flatnonzero(t_interp == t)[0]] for t in t])
    
    cmap = mpl.colormaps['jet'] #'viridis'
    if rango_T_fijo==True:
        norm_T = (np.array(T) - (-200)) / (50 - (-200))
    else:
        norm_T = (np.array(T) - np.array(T).min()) / (np.array(T).max() - np.array(T).min())
    
    colors = cmap(norm_T)
    
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    ax.plot(t_full,T_full,'.-',label=paths_T[0].split('/')[-1])
    ax.plot(t_interp,T_interp,'-',label='Temperatura interpolada')
    ax.scatter(t,T,color=colors,label='Temperatura muestra')

    plt.xlabel('t (s)')
    plt.ylabel('T (°C)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Temperatura de la muestra',fontsize=18)
    #plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_templog.png'),dpi=300,facecolor='w')
    plt.show()
    
    # Ajustar tiempos para que arranquen desde 0
    t = t - t_interp[0]
    t_interp = t_interp - t_interp[0]
    return t, T, t_interp, T_interp , colors,fig
    
#%% FF 
# directorio_FF = os.path.join(os.getcwd(),'C1','241010_154839') 
# paths_m_FF = glob(os.path.join(directorio_FF, '*.txt'))
# paths_m_FF.sort()
# paths_T_FF = glob(os.path.join(directorio_FF, '*templog*'))
# # levanto fecha de archivos grabada en meta 
# Fechas_FF = []

# for fp in paths_m_FF:
#     with open(fp, 'r') as f:
#         fecha_in_file = f.readline()
#         Fechas_FF.append(fecha_in_file.split()[-1])

# timestamp_FF,temperatura_FF,__ = lector_templog(paths_T_FF[0])

# t_full_FF = np.array([(t-timestamp_FF[0]).total_seconds() for t in timestamp_FF])
# T_full_FF= temperatura_FF
# dates_FF = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas_FF[:-1]] #datetimes c/ fecha de archivos 
# time_delta_FF=[t.total_seconds() for t in np.diff(dates_FF)] #dif de tiempo entre archivos c resolucion 0.01 s
# time_delta_FF.insert(0,0)
# delta_0_FF = (dates_FF[0] - timestamp_FF[0]).total_seconds() # entre comienzo del templog y 1er archivo redondeado a .2f

# #busco el indice en el templog que corresponde al segundo del 1er y ultimo dato para extrapolar tiempo y Temperatura 
# indx_1er_dato_FF=np.nonzero(timestamp_FF==dates_FF[0].replace(microsecond=0))[0][0]
# indx_ultimo_dato_FF=np.nonzero(timestamp_FF==datetime.strptime(Fechas_FF[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]
# #Interpolo t entre tiempo de 1er y ultimo ciclo 
# interp_func = interp1d(t_full_FF, T_full_FF, kind='linear')
# t_interp_FF = np.round(np.arange(t_full_FF[indx_1er_dato_FF], t_full_FF[indx_ultimo_dato_FF]+1.01,0.01),2)
# T_interp_FF= np.round(interp_func(t_interp_FF),2)

# t_FF = np.round(delta_0_FF + np.cumsum(time_delta_FF),2)
# T_FF = np.array([T_interp_FF[np.flatnonzero(t_interp_FF==t)[0]] for t in t_FF])

# cmap = mpl.colormaps['jet'] #'viridis'
# norm_T_FF = (np.array(T_FF) - np.array(T_FF).min()) / (np.array(T_FF).max() - np.array(T_FF).min())
# colors_FF = cmap(norm_T_FF)

# fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
# ax.plot(t_full_FF,T_full_FF,'.-',label='Templog (Rugged O201)')
# ax.plot(t_interp_FF,T_interp_FF,'-',label='Temperatura interpolada')
# ax.scatter(t_FF,T_FF,color=colors_FF,label='Temperatura muestra')

# plt.xlabel('t (s)')
# plt.ylabel('T (°C)')
# plt.legend(loc='lower right')
# plt.grid()
# plt.title('Temperatura de la muestra',fontsize=18)
# #plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_templog.png'),dpi=300,facecolor='w')
# plt.show()
# #%% SV
# directorio_SV = os.path.join(os.getcwd(),'SV','241010_151832') 
# paths_m_SV = glob(os.path.join(directorio_SV, '*.txt'))
# paths_m_SV.sort()
# paths_T_SV = glob(os.path.join(directorio_SV, '*templog*'))
# Fechas_SV=[]
# for fp in paths_m_SV:
#     with open(fp, 'r') as f:
#         fecha_in_file = f.readline()
#         Fechas_SV.append(fecha_in_file.split()[-1])
# timestamp_SV,temperatura_SV,__ = lector_templog(paths_T_SV[0])
# t_full_SV = np.array([(t-timestamp_SV[0]).total_seconds() for t in timestamp_SV])
# T_full_SV= temperatura_SV
# dates_SV = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas_SV[:-1]] #datetimes c/ fecha de archivos 
# time_delta=[t.total_seconds() for t in np.diff(dates_SV)] #dif de tiempo entre archivos c resolucion 0.01 s
# time_delta.insert(0,0)
# delta_0 = (dates_SV[0] - timestamp_SV[0]).total_seconds() # entre comienzo del templog y 1er archivo redondeado a .2f
# #busco el indice en el templog que corresponde al segundo del 1er y ultimo dato para extrapolar tiempo y Temperatura 
# indx_1er_dato=np.nonzero(timestamp_SV==dates_SV[0].replace(microsecond=0))[0][0]
# indx_ultimo_dato=np.nonzero(timestamp_SV==datetime.strptime(Fechas_SV[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]
# #Interpolo t entre tiempo de 1er y ultimo ciclo 
# interp_func = interp1d(t_full_SV, T_full_SV, kind='linear')
# t_interp_SV = np.round(np.arange(t_full_SV[indx_1er_dato], t_full_SV[indx_ultimo_dato]+1.01,0.01),2)
# T_interp_SV= np.round(interp_func(t_interp_SV),2)

# time_SV = np.round(delta_0 + np.cumsum(time_delta),2)
# T_SV = np.array([T_interp_SV[np.flatnonzero(t_interp_SV==t)[0]] for t in time_SV])

# cmap = mpl.colormaps['jet'] #'viridis'
# norm_T_SV = (np.array(T_SV) - np.array(T_SV).min()) / (np.array(T_SV).max() - np.array(T_SV).min())
# colors_SV = cmap(norm_T_SV)

# fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
# ax.plot(t_full_SV,T_full_SV,'.-',label='Templog (Rugged O201)')
# ax.plot(t_interp_SV,T_interp_SV,'-',label='Temperatura interpolada')
# ax.scatter(time_SV,T_SV,color=colors_SV,label='Temperatura muestra')

# plt.xlabel('t (s)')
# plt.ylabel('T (°C)')
# plt.legend(loc='lower right')
# plt.grid()
# plt.title('Temperatura de la muestra',fontsize=18)
# plt.show()


#%% Implementacion
directorios_FF = [os.path.join(os.getcwd(),'C1',f) for f in os.listdir('C1') ]
directorios_FF.sort()
t_FF_1,T_FF_1,t_interp_FF_1,T_interp_FF_1,c_FF_1,_=procesar_temperatura(directorios_FF[0])
t_FF_2,T_FF_2,t_interp_FF_2,T_interp_FF_2,c_FF_2,_=procesar_temperatura(directorios_FF[1])
t_FF_3,T_FF_3,t_interp_FF_3,T_interp_FF_3,c_FF_3,_=procesar_temperatura(directorios_FF[2])
t_FF_4,T_FF_4,t_interp_FF_4,T_interp_FF_4,c_FF_4,_=procesar_temperatura(directorios_FF[3])

directorios_SV = [os.path.join(os.getcwd(),'SV',f) for f in os.listdir('SV') ]
directorios_SV.sort()
t_SV_1,T_SV_1,t_interp_SV_1,T_interp_SV_1,c_SV_1,_=procesar_temperatura(directorios_SV[0])
t_SV_2,T_SV_2,t_interp_SV_2,T_interp_SV_2,c_SV_2,_=procesar_temperatura(directorios_SV[1])
t_SV_3,T_SV_3,t_interp_SV_3,T_interp_SV_3,c_SV_3,_=procesar_temperatura(directorios_SV[2])

# %% Derivadas
dT_SV_1 = np.gradient(T_SV_1,t_SV_1)
dT_SV_2 = np.gradient(T_SV_2,t_SV_2)
dT_SV_3 = np.gradient(T_SV_3,t_SV_3)
indx_max_SV_1 = np.nonzero(dT_SV_1==max(dT_SV_1))
indx_max_SV_2 = np.nonzero(dT_SV_2==max(dT_SV_2))
indx_max_SV_3 = np.nonzero(dT_SV_3==max(dT_SV_3))

dT_FF_1 = np.gradient(T_FF_1,t_FF_1)
dT_FF_2 = np.gradient(T_FF_2,t_FF_2)
dT_FF_3 = np.gradient(T_FF_3,t_FF_3)
dT_FF_4 = np.gradient(T_FF_4,t_FF_4)
indx_max_FF_1 = np.nonzero(dT_FF_1==max(dT_FF_1))
indx_max_FF_2 = np.nonzero(dT_FF_2==max(dT_FF_2))
indx_max_FF_3 = np.nonzero(dT_FF_3==max(dT_FF_3))
indx_max_FF_4 = np.nonzero(dT_FF_4==max(dT_FF_4))

#%% Grafico SV y FF+SV
fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(11,8),constrained_layout=True)

ax.plot(t_interp_SV_1,T_interp_SV_1,'-',label='T interp SV_1')
ax.plot(t_interp_SV_2,T_interp_SV_2,'-',label='T interp SV_2')
ax.plot(t_interp_SV_3,T_interp_SV_3,'-',label='T interp SV_3')

ax.scatter(t_SV_1,T_SV_1,color=c_SV_1,label='T SV_1',marker='.')
ax.scatter(t_SV_2,T_SV_2,color=c_SV_2,label='T SV_2',marker='.')
ax.scatter(t_SV_3,T_SV_3,color=c_SV_3,label='T SV_3',marker='.')

ax.scatter(t_SV_1[indx_max_SV_1],T_SV_1[indx_max_SV_1],marker='D',zorder=2,color='blue',label=f'dT/dt = {max(dT_SV_1):.2f} ºC/s')
ax.scatter(t_SV_2[indx_max_SV_2],T_SV_2[indx_max_SV_2],marker='D',zorder=2,color='orange',label=f'dT/dt = {max(dT_SV_2):.2f} ºC/s')
ax.scatter(t_SV_3[indx_max_SV_3],T_SV_3[indx_max_SV_3],marker='D',zorder=2,color='green',label=f'dT/dt = {max(dT_SV_3):.2f} ºC/s')

axin = ax.inset_axes([0.35, 0.15, 0.64, 0.58])
axin.plot(t_SV_1,dT_SV_1,'.-',lw=0.7,label='dT/dt SV_1')
axin.plot(t_SV_2,dT_SV_2,'.-',lw=0.7,label='dT/dt SV_2')
axin.plot(t_SV_3,dT_SV_3,'.-',lw=0.7,label='dT/dt SV_3')

ax2.plot(t_interp_FF_1,T_interp_FF_1,'-',label='T interp FF_1')
ax2.plot(t_interp_FF_3,T_interp_FF_3,'-',label='T interp FF_3')
ax2.plot(t_interp_FF_2,T_interp_FF_2,'-',label='T interp FF_2')
ax2.plot(t_interp_FF_4,T_interp_FF_4,'-',label='T interp FF_4')

ax2.scatter(t_FF_1,T_FF_1,color=c_FF_1,marker='.',label='T FF_1')
ax2.scatter(t_FF_2,T_FF_2,color=c_FF_2,marker='.',label='T FF_2')
ax2.scatter(t_FF_3,T_FF_3,color=c_FF_3,marker='.',label='T FF_3')
ax2.scatter(t_FF_4,T_FF_4,color=c_FF_4,marker='.',label='T FF_4')

ax2.scatter(t_FF_1[indx_max_FF_1],T_FF_1[indx_max_FF_1],marker='D',zorder=2,color='blue',label=f'dT/dt = {max(dT_FF_1):.2f} ºC/s')
ax2.scatter(t_FF_2[indx_max_FF_2],T_FF_2[indx_max_FF_2],marker='D',zorder=2,color='orange',label=f'dT/dt = {max(dT_FF_2):.2f} ºC/s')
ax2.scatter(t_FF_3[indx_max_FF_3],T_FF_3[indx_max_FF_3],marker='D',zorder=2,color='green',label=f'dT/dt = {max(dT_FF_3):.2f} ºC/s')

axin2 = ax2.inset_axes([0.35, 0.15, 0.64, 0.50])
axin2.plot(t_FF_1,dT_FF_1,'.-',lw=0.7,label='dT/dt FF_1')
axin2.plot(t_FF_2,dT_FF_2,'.-',lw=0.7,label='dT/dt FF_2')
axin2.plot(t_FF_3,dT_FF_3,'.-',lw=0.7,label='dT/dt FF_3')
axin2.plot(t_FF_4,dT_FF_4,'.-',lw=0.7,label='dT/dt FF_4')
for ai in [axin,axin2]:
    ai.set_xlabel('t (s)')
    ai.set_ylabel('dT/dt (ºC/s)')
    ai.legend(ncol=1)
    ai.grid()

# ax2.plot(t_interp_FF_1,cambio_T_interp_FF_1)
# ax2.plot(t_interp_FF_2,cambio_T_interp_FF_2)
# ax2.plot(t_interp_FF_3,cambio_T_interp_FF_3)
# ax2.plot(t_interp_FF_4,cambio_T_interp_FF_4)

for a in [ax,ax2]:
    a.grid()
    a.set_xlim(0,)
    a.set_ylabel('T (°C)')

ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

ax.set_title('SV',loc='left')
ax2.set_title('SV + NP',loc='left')
ax2.set_xlabel('t (s)')
plt.show()


#%%  Derivadas de las interpolaciones 
dT_interp_SV_1 = np.gradient(T_interp_SV_1,t_interp_SV_1)
dT_interp_SV_2 = np.gradient(T_interp_SV_2,t_interp_SV_2)
dT_interp_SV_3 = np.gradient(T_interp_SV_3,t_interp_SV_3)


#pruebo derivar, y luego interpolar
def indice_no_creciente(arr):
    for i in range(1, len(arr)):
        if arr[i] <= arr[i - 1]:
            return i  # Retorna el índice donde la secuencia deja de ser estrictamente creciente
    return -1  # Retorna -1 si toda la secuencia es estrictamente creciente


def indices_estrictamente_crecientes(arr):
    indices = []
    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            indices.append(i)
    return indices

indx_1=indices_estrictamente_crecientes(T_FF_1)
indx_2=indices_estrictamente_crecientes(T_FF_2)
indx_3=indices_estrictamente_crecientes(T_FF_3)

T_FF_1=T_FF_1[indx_1]
dT_FF_1=dT_FF_1[indx_1]

T_FF_2=T_FF_2[indx_2]
dT_FF_2=dT_FF_2[indx_2]

T_FF_3=T_FF_3[indx_3]
dT_FF_3=dT_FF_3[indx_3]


func_interp_1=PchipInterpolator(T_FF_1,dT_FF_1)
func_interp_2=PchipInterpolator(T_FF_2,dT_FF_2)
func_interp_3=PchipInterpolator(T_FF_3,dT_FF_3)

dT_interp_1_new=func_interp_1(T_SV_3)
dT_interp_2_new=func_interp_2(T_SV_3)
dT_interp_3_new=func_interp_3(T_SV_3)

resta_1= dT_interp_1_new-dT_SV_3
resta_2= dT_interp_2_new-dT_SV_3
resta_3= dT_interp_3_new-dT_SV_3

#%%
fig,(ax,ax2,ax3)=plt.subplots(nrows=3,figsize=(10,9),constrained_layout=True,sharex=True)

ax.plot(T_SV_3,dT_SV_3,'.-',label='dT SV_3',zorder=2)
ax.plot(T_FF_1,dT_FF_1,'.-',label='dT FF_1',zorder=2)


ax2.plot(T_SV_3,dT_SV_3,'.-',label='dT SV_3',zorder=3)
ax2.plot(T_FF_2,dT_FF_2,'.-',label='dT FF_2',zorder=1)

ax3.plot(T_FF_3,dT_FF_3,'.-',label='dT FF_3',zorder=2)
ax3.plot(T_SV_3,dT_SV_3,'o-',label='dT  SV_3',zorder=1)

# ax3.plot(T_SV_3,dT_interp_1_new,'.-',label='dT FF_test',zorder=2)
# ax3.plot(T_SV_3,resta_test,'.-',label='resta_1',zorder=2)

ax.set_title('dT/dt vs T  -  SV',loc='left')
ax2.set_title('dT/dt vs T -  SV intepolado',loc='left')

ax2.set_xlabel('T (°C)')

for a in [ax,ax2,ax3]:
    a.legend()
plt.show()














#%%veo las derivadas ahora que descontamos el calentamiento por atmosfera

fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)

ax.plot(t_FF_1,T_FF_1,'.-',label='T FF_1')
ax.plot(t_FF_2,T_FF_2,'.-',label='T FF_2')
ax.plot(t_FF_3,T_FF_3,'.-',label='T FF_3')
ax.plot(t_FF_4,T_FF_4,'.-',label='T FF_4')
ax.plot(t_SV_1,T_SV_1,'.-',label='T SV_1')

ax2.plot(t_FF_1,cambio_T_FF_1,'.-',label='$\Delta$T FF_1')
ax2.plot(t_FF_2,cambio_T_FF_2,'.-',label='$\Delta$T FF_2')
ax2.plot(t_FF_3,cambio_T_FF_3,'.-',label='$\Delta$T FF_3')
ax2.plot(t_FF_4,cambio_T_FF_4,'.-',label='$\Delta$T FF_4')
for a in [ax,ax2]:
    a.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    a.grid()
    a.set_xlim(0,160)

    #a.set_xlabel('t (s)')
ax2.set_ylabel('$\Delta$T (°C)')
ax.set_title('Temperatura de la muestra',loc='left')
ax2.set_title('$\Delta$T por NP',loc='left')
ax2.set_xlabel('t (s)')
plt.savefig('cambio_enT_porNP.png',dpi=300)
plt.show()

#%% derivo el cambbio de T para obtener SAR 'calorimetrico'

dCambio_FF_1 = np.gradient(cambio_T_FF_1,t_FF_1)
dCambio_FF_2 = np.gradient(cambio_T_FF_2,t_FF_2)
dCambio_FF_3 = np.gradient(cambio_T_FF_3,t_FF_3)
dCambio_FF_4 = np.gradient(cambio_T_FF_4,t_FF_4)

indx_TF_1= np.nonzero(T_FF_1<=0) 
indx_TF_2= np.nonzero(T_FF_2<=0) 
indx_TF_3= np.nonzero(T_FF_3<=0) 
indx_TF_4= np.nonzero(T_FF_4<=0) 


c_liquido = 4.182 #J/gK 

C1=15/1000 #15g/L / 1L/1000g


# SAR_FF_2 =  dCambio_FF_2/C1
# SAR_FF_2[:indx_TF_2[0][-1]]*=c_solido
# SAR_FF_2[indx_TF_2[0][-1]:]*=c_liquido

# SAR_FF_3 =  dCambio_FF_3/C1
# SAR_FF_3[:indx_TF_3[0][-1]]*=c_solido
# SAR_FF_3[indx_TF_3[0][-1]:]*=c_liquido

# SAR_FF_4 =  dCambio_FF_4/C1
# SAR_FF_4[:indx_TF_4[0][-1]]*=c_solido
# SAR_FF_4[indx_TF_4[0][-1]:]*=c_liquido


#%% Para temperaturas bajo 0

T= np.array([80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,273])-273
cp= np.array([711.48,794.93,874.14,949.38,1021.30,1090.80,1158.82,1226.18,1293.51,1361.21,1429.53,1498.57,1568.35,
     1638.86,1710.03,1781.79,1854.08,1926.83,1999.98,2073.48,2095.59])/1000
interp_func_aux = splrep(T,cp)
Cp_aux_FF_1=splev(T_FF_1[indx_TF_1[0]],interp_func_aux)
Cp_aux_FF_2=splev(T_FF_2[indx_TF_2[0]],interp_func_aux)
Cp_aux_FF_3=splev(T_FF_3[indx_TF_3[0]],interp_func_aux)
Cp_aux_FF_4=splev(T_FF_4[indx_TF_4[0]],interp_func_aux)
# %%
SAR_FF_1 =  dCambio_FF_1/C1
SAR_FF_1[:indx_TF_1[0][-1]+1]*=Cp_aux_FF_1
SAR_FF_1[indx_TF_1[0][-1]+1:]*=c_liquido

SAR_FF_2 =  dCambio_FF_2/C1
SAR_FF_2[:indx_TF_2[0][-1]+1]*=Cp_aux_FF_2
SAR_FF_2[indx_TF_2[0][-1]+1:]*=c_liquido

SAR_FF_3 =  dCambio_FF_3/C1
SAR_FF_3[:indx_TF_3[0][-1]+1]*=Cp_aux_FF_3
SAR_FF_3[indx_TF_3[0][-1]+1:]*=c_liquido

SAR_FF_4 =  dCambio_FF_4/C1
SAR_FF_4[:indx_TF_4[0][-1]+1]*=Cp_aux_FF_4
SAR_FF_4[indx_TF_4[0][-1]+1:]*=c_liquido








fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)

ax.plot(t_FF_1,SAR_FF_1,'.-',label='SAR FF_1')
ax.plot(t_FF_2,SAR_FF_2,'.-',label='SAR FF_2')
ax.plot(t_FF_3,SAR_FF_3,'.-',label='SAR FF_3')
ax.plot(t_FF_4,SAR_FF_4,'.-',label='SAR FF_4')


ax.set_title('SAR vs t',loc='left')
ax.grid()
ax.set_xlabel('t (s)')
ax.set_ylabel('SAR (W/g)')
plt.savefig('sar_vs_t.png',dpi=300)
plt.show()
# %%
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)

ax.plot(T_FF_1,SAR_FF_1,'.-',label='SAR FF_1')
ax.plot(T_FF_2,SAR_FF_2,'.-',label='SAR FF_2')
ax.plot(T_FF_3,SAR_FF_3,'.-',label='SAR FF_3')
ax.plot(T_FF_4,SAR_FF_4,'.-',label='SAR FF_4')


ax.set_title('SAR vs Temp',loc='left')
ax.grid()
ax.set_xlabel('T (°C)')
ax.set_ylabel('SAR (W/g)')
plt.savefig('sar_vs_Temp.png',dpi=300)
plt.show()
# %%

dT_interp_FF_1 = np.gradient(T_interp_FF_1,t_interp_FF_1)
dT_interp_FF_2 = np.gradient(T_interp_FF_2,t_interp_FF_2)
dT_interp_FF_3 = np.gradient(T_interp_FF_3,t_interp_FF_3)
dT_interp_FF_4 = np.gradient(T_interp_FF_4,t_interp_FF_4)


fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)

ax.plot(T_FF_1,dT_FF_1,'.-',label='dT FF_1')
ax.plot(T_FF_2,dT_FF_2,'.-',label='dT FF_2')
ax.plot(T_FF_3,dT_FF_3,'.-',label='dT FF_3')

ax2.plot(T_interp_FF_1,dT_interp_FF_1,'.-',alpha=0.7,label='dT interp FF_1')
ax2.plot(T_interp_FF_2,dT_interp_FF_2,'.-',label='dT interp FF_2')
ax2.plot(T_interp_FF_3,dT_interp_FF_3,'.-',label='dT interp FF_3')

ax.set_title('dT/dt vs T  -  FF',loc='left')
ax2.set_title('dT/dt vs T -  FF intepolado',loc='left')

ax2.set_xlabel('T (°C)')
plt.legend()
plt.show()
#%% Resto a todos el templog del SV 1 interpolando 

interp_func_1 = PchipInterpolator(t_SV_1,T_SV_1)
T_aux_FF_1=interp_func_1(t_FF_1)
T_aux_FF_2=interp_func_1(t_FF_2)
T_aux_FF_3=interp_func_1(t_FF_3)
T_aux_FF_4=interp_func_1(t_FF_4)

cambio_T_FF_1 = T_FF_1 - T_aux_FF_1
cambio_T_FF_2 = T_FF_2 - T_aux_FF_2
cambio_T_FF_3 = T_FF_3 - T_aux_FF_3
cambio_T_FF_4 = T_FF_4 - T_aux_FF_4


fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(10,6),constrained_layout=True)

ax.plot(t_FF_1,T_FF_1,'.-',label='T FF_1')
ax.plot(t_FF_2,T_FF_2,'.-',label='T FF_2')
ax.plot(t_FF_3,T_FF_3,'.-',label='T FF_3')
ax.plot(t_FF_4,T_FF_4,'.-',label='T FF_4')

ax.plot(t_SV_1,T_SV_1,'.-',label='T SV_1')

ax2.plot(t_FF_1,cambio_T_FF_1,'.-',label='$\Delta$T FF_1')
ax2.plot(t_FF_2,cambio_T_FF_2,'.-',label='$\Delta$T FF_2')
ax2.plot(t_FF_3,cambio_T_FF_3,'.-',label='$\Delta$T FF_3')
ax2.plot(t_FF_4,cambio_T_FF_4,'.-',label='$\Delta$T FF_4')
for a in [ax,ax2]:
    a.legend(loc='lower right',ncol=2)
    a.grid()
    a.set_xlim(0,160)  
    
    #a.set_xlabel('t (s)')
ax2.set_ylabel('$\Delta$T (°C)')
ax.set_title('Temperatura de la muestra',loc='left')
ax2.set_title('Cambio en Temperatura por NP',loc='left')
ax2.set_xlabel('t (s)')
plt.savefig('cambio_en_T_por_NP.png',dpi=300)
plt.show()

#%% derivo el cambbio de T para obtener SAR 'calorimetrico'

dCambio_FF_1 = np.gradient(cambio_T_FF_1,t_FF_1)
dCambio_FF_2 = np.gradient(cambio_T_FF_2,t_FF_2)
dCambio_FF_3 = np.gradient(cambio_T_FF_3,t_FF_3)
dCambio_FF_4 = np.gradient(cambio_T_FF_4,t_FF_4)

indx_TF_1= np.nonzero(T_FF_1<=0) 
indx_TF_2= np.nonzero(T_FF_2<=0) 
indx_TF_3= np.nonzero(T_FF_3<=0) 
indx_TF_4= np.nonzero(T_FF_4<=0) 


c_liquido = 4.182 #J/gK 

C1=15/1000 #15g/L / 1L/1000g

#%% Para temperaturas bajo 0

T= np.array([80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,273])-273
cp= np.array([711.48,794.93,874.14,949.38,1021.30,1090.80,1158.82,1226.18,1293.51,1361.21,1429.53,1498.57,1568.35,
     1638.86,1710.03,1781.79,1854.08,1926.83,1999.98,2073.48,2095.59])/1000

interp_func_cp = PchipInterpolator(T,cp)
Cp_aux_FF_1=interp_func_cp(T_FF_1[indx_TF_1[0]])
Cp_aux_FF_2=interp_func_cp(T_FF_2[indx_TF_2[0]])
Cp_aux_FF_3=interp_func_cp(T_FF_3[indx_TF_3[0]])
Cp_aux_FF_4=interp_func_cp(T_FF_4[indx_TF_4[0]])
# %%
SAR_FF_1 =  dCambio_FF_1/C1
SAR_FF_1[:indx_TF_1[0][-1]+1]*=Cp_aux_FF_1
SAR_FF_1[indx_TF_1[0][-1]+1:]*=c_liquido

SAR_FF_2 =  dCambio_FF_2/C1
SAR_FF_2[:indx_TF_2[0][-1]+1]*=Cp_aux_FF_2
SAR_FF_2[indx_TF_2[0][-1]+1:]*=c_liquido

SAR_FF_3 =  dCambio_FF_3/C1
SAR_FF_3[:indx_TF_3[0][-1]+1]*=Cp_aux_FF_3
SAR_FF_3[indx_TF_3[0][-1]+1:]*=c_liquido

SAR_FF_4 =  dCambio_FF_4/C1
SAR_FF_4[:indx_TF_4[0][-1]+1]*=Cp_aux_FF_4
SAR_FF_4[indx_TF_4[0][-1]+1:]*=c_liquido


fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)

ax.plot(t_FF_1,SAR_FF_1,'.-',label='SAR FF_1')
ax.plot(t_FF_2,SAR_FF_2,'.-',label='SAR FF_2')
ax.plot(t_FF_3,SAR_FF_3,'.-',label='SAR FF_3')
ax.plot(t_FF_4,SAR_FF_4,'.-',label='SAR FF_4')

ax.set_title('SAR vs t',loc='left')
ax.grid()
ax.set_xlabel('t (s)')
ax.set_ylabel('SAR (W/g)')
plt.savefig('sar_vs_t.png',dpi=300)
plt.show()


fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
ax.plot(T_FF_1,SAR_FF_1,'.-',label='SAR FF_1')
ax.plot(T_FF_2,SAR_FF_2,'.-',label='SAR FF_2')
ax.plot(T_FF_3,SAR_FF_3,'.-',label='SAR FF_3')
ax.plot(T_FF_4,SAR_FF_4,'.-',label='SAR FF_4')

ax.set_title('SAR vs Temp',loc='left')
ax.grid()
ax.set_xlabel('T (°C)')
ax.set_ylabel('SAR (W/g)')
plt.savefig('sar_vs_Temp.png',dpi=300)
plt.show()

# %% Ahora con SAR EM 

indices_temp_300_150_C1_1 = get_indices_by_range(T_300_C1_1,temperature_ranges_all)
indices_temp_300_150_C1_2 = get_indices_by_range(T_300_C1_2,temperature_ranges_all)
indices_temp_300_150_C1_3 = get_indices_by_range(T_300_C1_3,temperature_ranges_all)
indices_temp_300_150_C1_4 = get_indices_by_range(T_300_C1_4,temperature_ranges_all)
# Lista de listas de índices por archivo de temperatura
indices_by_temp = [indices_temp_300_150_C1_2, indices_temp_300_150_C1_3, indices_temp_300_150_C1_4]

Temp_300_C1 = []
Temp_300_C1_err = []
SAR_300_C1 = []
SAR_300_C1_err = []
# Cálculo de promedios y desviaciones estándar por rango de temperatura
for i in range(len(temperature_ranges_all)):
    # Promedio para T_300
    Temp_300_C1.append(np.mean(np.concatenate([T_300_C1_2[indices_temp_300_150_C1_2[i]],
                                            T_300_C1_3[indices_temp_300_150_C1_3[i]],
                                            T_300_C1_4[indices_temp_300_150_C1_4[i]]])))

    Temp_300_C1_err.append(np.std(np.concatenate([T_300_C1_2[indices_temp_300_150_C1_2[i]],
                                                T_300_C1_3[indices_temp_300_150_C1_3[i]],
                                                T_300_C1_4[indices_temp_300_150_C1_4[i]]]))),

    SAR_300_C1.append(np.mean(np.concatenate([SAR_300_C1_2[indices_temp_300_150_C1_2[i]],
                                           SAR_300_C1_3[indices_temp_300_150_C1_3[i]],
                                           SAR_300_C1_4[indices_temp_300_150_C1_4[i]]])))

    SAR_300_C1_err.append(np.std(np.concatenate([SAR_300_C1_2[indices_temp_300_150_C1_2[i]],
                                                 SAR_300_C1_3[indices_temp_300_150_C1_3[i]],
                                            SAR_300_C1_4[indices_temp_300_150_C1_4[i]]])))

#remuevo elementos nan
Temp_300_C1 = [i for i in Temp_300_C1 if ~np.isnan(i)]
Temp_300_C1_err = [i for i in Temp_300_C1_err if ~np.isnan(i)]
SAR_300_C1 = [i for i in SAR_300_C1 if ~np.isnan(i)]
SAR_300_C1_err = [i for i in SAR_300_C1_err if ~np.isnan(i)]


fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(11,7),constrained_layout=True,sharex=True)
ax.set_title('All',loc='left')
ax.plot(T_300_C1_1,SAR_300_C1_1,'-',label='C1')
ax.plot(T_300_C1_2,SAR_300_C1_2,'-',label='C1')
ax.plot(T_300_C1_3,SAR_300_C1_3,'-',label='C1')
ax.plot(T_300_C1_4,SAR_300_C1_4,'-',label='C1')


ax2.set_title('Promedios',loc='left')
ax2.errorbar(x=Temp_300_C1,y=SAR_300_C1,xerr=Temp_300_C1_err,yerr=SAR_300_C1_err,fmt='.-',capsize=3,label='C1')
# ax2.errorbar(x=Temp_300_C2,y=SAR_300_C2,xerr=Temp_300_C2_err,yerr=SAR_300_C2_err,fmt='.-',capsize=3,label='C2')
ax2.set_xlabel('T (°C)')


for a in [ax,ax2]:
    a.grid()
    a.set_ylabel('SAR (W/g)')
    a.legend(title=f'''C1 = {meta_C1_2["Concentracion g/m^3"]/1e3:.1f} g/L''',ncol=2)
plt.suptitle(f'SAR vs T\nNE5X en SV\n$f$ = 300 kHz    $H_0$ = 57 kA/m',fontsize=13)
#plt.savefig('C1_C2_comparacion.png',dpi=300)
#%%

indices_temp_FF_1 = get_indices_by_range(T_FF_1,temperature_ranges_all)
indices_temp_FF_2 = get_indices_by_range(T_FF_2,temperature_ranges_all)
indices_temp_FF_3 = get_indices_by_range(T_FF_3,temperature_ranges_all)
indices_temp_FF_4 = get_indices_by_range(T_FF_4,temperature_ranges_all)
# Lista de listas de índices por archivo de temperatura
indices_by_temp = [indices_temp_FF_1,indices_temp_FF_2, indices_temp_FF_3, indices_temp_FF_4]

Temp_FF = []
Temp_FF_err = []
SAR_FF = []
SAR_FF_err = []
# Cálculo de promedios y desviaciones estándar por rango de temperatura
for i in range(len(temperature_ranges_all)):
    # Promedio para T_300
    Temp_FF.append(np.mean(np.concatenate([T_FF_1[indices_temp_FF_1[i]],
                                            T_FF_2[indices_temp_FF_2[i]],
                                            T_FF_3[indices_temp_FF_3[i]],
                                            T_FF_4[indices_temp_FF_4[i]]])))

    Temp_FF_err.append(np.std(np.concatenate([T_FF_1[indices_temp_FF_1[i]],
                                            T_FF_2[indices_temp_FF_2[i]],
                                                T_FF_3[indices_temp_FF_3[i]],
                                                T_FF_4[indices_temp_FF_4[i]]]))),

    SAR_FF.append(np.mean(np.concatenate([SAR_FF_1[indices_temp_FF_1[i]],
                                        SAR_FF_2[indices_temp_FF_2[i]],
                                           SAR_FF_3[indices_temp_FF_3[i]],
                                           SAR_FF_4[indices_temp_FF_4[i]]])))

    SAR_FF_err.append(np.std(np.concatenate([SAR_FF_1[indices_temp_FF_1[i]],
                                        SAR_FF_2[indices_temp_FF_2[i]],
                                            SAR_FF_3[indices_temp_FF_3[i]],
                                            SAR_FF_4[indices_temp_FF_4[i]]])))

#remuevo elementos nan
Temp_FF = [i for i in Temp_FF if ~np.isnan(i)]
Temp_FF_err = [i for i in Temp_FF_err if ~np.isnan(i)]
SAR_FF = [i for i in SAR_FF if ~np.isnan(i)]
SAR_FF_err = [i for i in SAR_FF_err if ~np.isnan(i)]




#%% Comparativa
fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(11,7),constrained_layout=True,sharex=True)
ax.set_title('All',loc='left')
ax.plot(T_300_C1_1,SAR_300_C1_1,'-',label='C1')
ax.plot(T_300_C1_2,SAR_300_C1_2,'-',label='C1')
ax.plot(T_300_C1_3,SAR_300_C1_3,'-',label='C1')
ax.plot(T_300_C1_4,SAR_300_C1_4,'-',label='C1')

ax.plot(T_FF_1,SAR_FF_1,'.-',label='SAR FF_1')
ax.plot(T_FF_2,SAR_FF_2,'.-',label='SAR FF_2')
ax.plot(T_FF_3,SAR_FF_3,'.-',label='SAR FF_3')
ax.plot(T_FF_4,SAR_FF_4,'.-',label='SAR FF_4')

# ax.plot(T_300_C2_1,SAR_300_C2_1,'.-',lw=0.7,label='C2')
# ax.plot(T_300_C2_2,SAR_300_C2_2,'.-',lw=0.7,label='C2')
# ax.plot(T_300_C2_3,SAR_300_C2_3,'.-',lw=0.7,label='C2')
# ax.plot(T_300_C2_4,SAR_300_C2_4,'.-',lw=0.7,label='C2')

ax2.set_title('Promedios',loc='left')
ax2.errorbar(x=Temp_300_C1,y=SAR_300_C1,xerr=Temp_300_C1_err,yerr=SAR_300_C1_err,fmt='.-',capsize=3,label='ESAR')
ax2.errorbar(x=Temp_FF,y=SAR_FF,xerr=Temp_FF_err,yerr=SAR_FF_err,fmt='.-',capsize=3,label='CSAR')
ax2.set_xlabel('T (°C)')


for a in [ax,ax2]:
    a.grid()
    a.set_ylabel('SAR (W/g)')
    a.legend(title=f'''C1 = {meta_C1_2["Concentracion g/m^3"]/1e3:.1f} g/L''',ncol=2)
plt.suptitle(f'SAR vs T\nNE5X en SV\n$f$ = 300 kHz    $H_0$ = 57 kA/m',fontsize=13)
plt.savefig('comparativa_ESAR_CSAR.png',dpi=300)
# %%
