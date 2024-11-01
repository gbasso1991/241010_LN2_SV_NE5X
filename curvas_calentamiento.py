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

def procesar_temperatura(directorio_FF,rango_T_fijo=(-200,50)):
    # Obtener archivos de datos y archivos de templog
    paths_m_FF = glob(os.path.join(directorio_FF, '*.txt'))
    paths_m_FF.sort()
    paths_T_FF = glob(os.path.join(directorio_FF, '*templog*'))

    # Levantar fechas de archivos grabadas en meta
    Fechas_FF = []
    for fp in paths_m_FF:
        with open(fp, 'r') as f:
            fecha_in_file = f.readline()
            Fechas_FF.append(fecha_in_file.split()[-1])

    # Obtener timestamps y temperaturas del templog
    timestamp_FF, temperatura_FF, __ = lector_templog(paths_T_FF[0])

    # Calcular tiempos completos en segundos
    t_full_FF = np.array([(t - timestamp_FF[0]).total_seconds() for t in timestamp_FF])
    T_full_FF = temperatura_FF

    # Procesar las fechas y tiempos de los archivos
    dates_FF = [datetime.strptime(f, '%y%m%d_%H:%M:%S.%f') for f in Fechas_FF[:-1]]  # datetimes con fecha de archivos
    time_delta_FF = [t.total_seconds() for t in np.diff(dates_FF)]  # diferencia de tiempo entre archivos
    time_delta_FF.insert(0, 0)  # Insertar el primer delta como 0
    delta_0_FF = (dates_FF[0] - timestamp_FF[0]).total_seconds()  # diferencia entre comienzo templog y 1er archivo

    # Buscar los índices de los datos de templog correspondientes al primer y último archivo
    indx_1er_dato_FF = np.nonzero(timestamp_FF == dates_FF[0].replace(microsecond=0))[0][0]
    indx_ultimo_dato_FF = np.nonzero(timestamp_FF == datetime.strptime(Fechas_FF[-1], '%y%m%d_%H:%M:%S.%f').replace(microsecond=0))[0][0]

    # Interpolación entre el primer y último ciclo
    interp_func = interp1d(t_full_FF, T_full_FF, kind='linear')
    t_interp_FF = np.round(np.arange(t_full_FF[indx_1er_dato_FF], t_full_FF[indx_ultimo_dato_FF] + 1.01, 0.01), 2)
    T_interp_FF = np.round(interp_func(t_interp_FF), 2)

    # Calcular t_FF y T_FF a partir de los datos
    t_FF = np.round(delta_0_FF + np.cumsum(time_delta_FF), 2)
    T_FF = np.array([T_interp_FF[np.flatnonzero(t_interp_FF == t)[0]] for t in t_FF])

    cmap = mpl.colormaps['jet'] #'viridis'
    if rango_T_fijo==True:
        norm_T_FF = (np.array(T_FF) - (rango_T_fijo[0])) / (rango_T_fijo[1] - rango_T_fijo[0])
        print(rango_T_fijo)
    else:
        norm_T_FF = (np.array(T_FF) - np.array(T_FF).min()) / (np.array(T_FF).max() - np.array(T_FF).min())

    colors_FF = cmap(norm_T_FF)

    
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    ax.plot(t_full_FF,T_full_FF,'.-',label=os.path.split(paths_T_FF[0])[1])
    ax.plot(t_interp_FF,T_interp_FF,'-',label='Temperatura interpolada')
    ax.scatter(t_FF,T_FF,color=colors_FF,label='Temperatura muestra')

    plt.xlabel('t (s)')
    plt.ylabel('T (°C)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.title('Temperatura de la muestra',fontsize=18)
    #plt.savefig(os.path.join(output_dir,os.path.commonprefix(fnames_m)+'_templog.png'),dpi=300,facecolor='w')
    plt.show()

    # Ajustar tiempos para que arranquen desde 0
    t_FF = t_FF - t_interp_FF[0]
    t_interp_FF = t_interp_FF - t_interp_FF[0]


    return t_FF, T_FF, t_interp_FF, T_interp_FF , colors_FF,fig
<<<<<<< HEAD
    
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

=======

# #%% FF
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

>>>>>>> 6677b983bdab0f75e04fb0ef9c540a15bcaf3999
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

<<<<<<< HEAD
#%% Resto a todos el templog del SV 1 interpolando 
from scipy.interpolate import splev  , splrep
=======
#%% Resto a todos el templog del SV 1 interpolando

from scipy.interpolate import splev,splrep
>>>>>>> 6677b983bdab0f75e04fb0ef9c540a15bcaf3999

interp_func_1 = splrep(t_SV_1,T_SV_1)
T_aux_FF_1=splev(t_FF_1,interp_func_1)
T_aux_FF_2=splev(t_FF_2,interp_func_1)
T_aux_FF_3=splev(t_FF_3,interp_func_1)
T_aux_FF_4=splev(t_FF_4,interp_func_1)

cambio_T_FF_1 = T_FF_1 - T_aux_FF_1
cambio_T_FF_2 = T_FF_2 - T_aux_FF_2
cambio_T_FF_3 = T_FF_3 - T_aux_FF_3
cambio_T_FF_4 = T_FF_4 - T_aux_FF_4
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
<<<<<<< HEAD
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
=======

plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
# Ahora los templogs del enfriamiento con Loc de fallas
directorios_FF_LF = [os.path.join(os.getcwd(),'LF',f) for f in os.listdir('LF') ]
directorios_FF_LF.sort()
t_LF_1,T_LF_1,t_interp_LF_1,T_interp_LF_1,c_LF_1,_=procesar_temperatura(directorios_FF_LF[0])
t_LF_2,T_LF_2,t_interp_LF_2,T_interp_LF_2,c_LF_2,_=procesar_temperatura(directorios_FF_LF[1],rango_T_fijo=(-200,100))
# t_FF_3,T_FF_3,t_interp_FF_3,T_interp_FF_3,c_FF_3,_=procesar_temperatura(directorios_FF[2])
# t_FF_4,T_FF_4,t_interp_FF_4,T_interp_FF_4,c_FF_4,_=procesar_temperatura(directorios_FF[3])


>>>>>>> 6677b983bdab0f75e04fb0ef9c540a15bcaf3999
# %%
