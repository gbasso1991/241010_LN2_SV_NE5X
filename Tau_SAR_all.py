#%% Analisis para calculo de tau - levantando de archivo resultados.txt y de ciclo
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
#%% LECTOR RESULTADOS
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=18,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N



#%% C1 300 57
resultados_C1_300 = glob(os.path.join('./C1', '**', '*resultados.txt'),recursive=True)
resultados_C1_300.sort()
meta_C1_1,files_1,time_1,T_300_C1_1,Mr_300_C1_1,Hc_300_C1_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_0_1,dphi_fem_0_1,SAR_300_C1_1,tau_300_C1_1,N1 = lector_resultados(resultados_C1_300[0])
meta_C1_2,files_2,time_2,T_300_C1_2,Mr_300_C1_2,Hc_300_C1_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_0_2,dphi_fem_0_2,SAR_300_C1_2,tau_300_C1_2,N2 = lector_resultados(resultados_C1_300[1])
meta_C1_3,files_3,time_3,T_300_C1_3,Mr_300_C1_3,Hc_300_C1_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_0_3,dphi_fem_0_3,SAR_300_C1_3,tau_300_C1_3,N3 = lector_resultados(resultados_C1_300[2])
meta_C1_4,files_4,time_4,T_300_C1_4,Mr_300_C1_4,Hc_300_C1_4,campo_max_4,mag_max_4,xi_M_0_4,frecuencia_fund_4,magnitud_fund_0_4,dphi_fem_0_4,SAR_300_C1_4,tau_300_C1_4,N3 = lector_resultados(resultados_C1_300[3])

#% Tau vs Temp
fig,((ax,ax2),(ax3,ax4))= plt.subplots(nrows=2,ncols=2,figsize=(12,6),constrained_layout=True,sharex=True)

ax.plot(T_300_C1_1,tau_300_C1_1,'.-',label='1')
ax.plot(T_300_C1_2,tau_300_C1_2,'.-',label='2')
ax.plot(T_300_C1_3,tau_300_C1_3,'.-',label='3')
ax.plot(T_300_C1_4,tau_300_C1_4,'.-',label='4')
ax.set_title(r'$\tau$')
ax.set_ylabel(r'$\tau$ (s)')

ax2.plot(T_300_C1_1,SAR_300_C1_1,'.-',label='1')
ax2.plot(T_300_C1_2,SAR_300_C1_2,'.-',label='2')
ax2.plot(T_300_C1_3,SAR_300_C1_3,'.-',label='3')
ax2.plot(T_300_C1_4,SAR_300_C1_4,'.-',label='4')
ax2.set_title('SAR')
ax2.set_ylabel('SAR (W/g)')

ax3.plot(T_300_C1_1,Mr_300_C1_1,'.-',label='1')
ax3.plot(T_300_C1_2,Mr_300_C1_2,'.-',label='2')
ax3.plot(T_300_C1_3,Mr_300_C1_3,'.-',label='3')
ax3.plot(T_300_C1_4,Mr_300_C1_4,'.-',label='4')
ax3.set_title('M$_R$')
ax3.set_ylabel('M$_R$ (A/m)')

ax4.plot(T_300_C1_1,Hc_300_C1_1,'.-',label='1')
ax4.plot(T_300_C1_2,Hc_300_C1_2,'.-',label='2')
ax4.plot(T_300_C1_3,Hc_300_C1_3,'.-',label='3')
ax4.plot(T_300_C1_4,Hc_300_C1_4,'.-',label='4')
ax4.set_title('H$_C$')
ax4.set_ylabel('H$_C$  (kA/m)')

for a in [ax,ax2,ax3,ax4]:
    a.legend()
    a.grid()
    
ax3.set_xlabel('T (°C)')
ax4.set_xlabel('T (°C)')
plt.suptitle('Comparativa NE5X - C1\n300 kHz 57 kA/m')
plt.savefig('C1_300_57_NE5X_comparativa.png',dpi=300)
plt.show()

#%% C2 300 57
resultados_C2_300 = glob(os.path.join('./C2', '**', '*resultados.txt'),recursive=True)
resultados_C2_300.sort()
meta_C2_1,files_1,time_1,T_300_C2_1,Mr_C2_1,Hc_C2_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_0_1,dphi_fem_0_1,SAR_300_C2_1,tau_300_C2_1,N1 = lector_resultados(resultados_C2_300[0])
meta_C2_2,files_2,time_2,T_300_C2_2,Mr_C2_2,Hc_C2_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_0_2,dphi_fem_0_2,SAR_300_C2_2,tau_300_C2_2,N2 = lector_resultados(resultados_C2_300[1])
meta_C2_3,files_3,time_3,T_300_C2_3,Mr_C2_3,Hc_C2_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_0_3,dphi_fem_0_3,SAR_300_C2_3,tau_300_C2_3,N3 = lector_resultados(resultados_C2_300[2])
meta_C2_4,files_4,time_4,T_300_C2_4,Mr_C2_4,Hc_C2_4,campo_max_4,mag_max_4,xi_M_0_4,frecuencia_fund_4,magnitud_fund_0_4,dphi_fem_0_4,SAR_300_C2_4,tau_300_C2_4,N3 = lector_resultados(resultados_C2_300[3])

#% Tau vs Temp
fig,((ax,ax2),(ax3,ax4))= plt.subplots(nrows=2,ncols=2,figsize=(12,6),constrained_layout=True,sharex=True)

ax.plot(T_300_C2_1,tau_300_C2_1,'.-',label='1')
ax.plot(T_300_C2_2,tau_300_C2_2,'.-',label='2')
ax.plot(T_300_C2_3,tau_300_C2_3,'.-',label='3')

ax.plot(T_300_C2_4,tau_300_C2_4,'.-',label='4')
ax.set_title(r'$\tau$')
ax.set_ylabel(r'$\tau$ (s)')

ax2.plot(T_300_C2_1,SAR_300_C2_1,'.-',label='1')
ax2.plot(T_300_C2_2,SAR_300_C2_2,'.-',label='2')
ax2.plot(T_300_C2_3,SAR_300_C2_3,'.-',label='3')
ax2.plot(T_300_C2_4,SAR_300_C2_4,'.-',label='4')
ax2.set_title('SAR')
ax2.set_ylabel('SAR (W/g)')

ax3.plot(T_300_C2_1,Mr_C2_1,'.-',label='1')
ax3.plot(T_300_C2_2,Mr_C2_2,'.-',label='2')
ax3.plot(T_300_C2_3,Mr_C2_3,'.-',label='3')
ax3.plot(T_300_C2_4,Mr_C2_4,'.-',label='4')
ax3.set_title('M$_R$')
ax3.set_ylabel('M$_R$ (A/m)')

ax4.plot(T_300_C2_1,Hc_C2_1,'.-',label='1')
ax4.plot(T_300_C2_2,Hc_C2_2,'.-',label='2')
ax4.plot(T_300_C2_3,Hc_C2_3,'.-',label='3')
ax4.plot(T_300_C2_4,Hc_C2_4,'.-',label='4')
ax4.set_title('H$_C$')
ax4.set_ylabel('H$_C$  (kA/m)')

for a in [ax,ax2,ax3,ax4]:
    a.legend()
    a.grid()
    
ax3.set_xlabel('T (°C)')
ax4.set_xlabel('T (°C)')
plt.suptitle('Comparativa NE5X - C2\n300 kHz 57 kA/m')
plt.savefig('C2_300_57_NE5X_comparativa.png',dpi=300)
plt.show()


#%% Promediado de valores en rangosde de temperatura 

# Definir los rangos de temperatura

temperature_ranges_1 = [(i, i + 5) for i in np.arange(-170,-10,5)]
print('TR1 = ',temperature_ranges_1)

temperature_ranges_2 = [(j, j + 0.5) for j in np.arange(-10,10,0.5)]
print('TR2 = ',temperature_ranges_2)
    
temperature_ranges_3 = [(k, k + 2) for k in np.arange(10,30,1)]
print('TR3 = ',temperature_ranges_3)

# Función para obtener los índices por rango de temperatura
def get_indices_by_range(temp_array, temperature_ranges):
    indices_by_range = []
    for range_min, range_max in temperature_ranges:
        indices = np.where((temp_array > range_min) & (temp_array <= range_max))
        indices_by_range.append(indices[0])  # Acceder a los índices reales
    return indices_by_range

temperature_ranges_all = temperature_ranges_1+temperature_ranges_2+temperature_ranges_3
#%% Promedios
# C1
indices_temp_300_150_C1_1 = get_indices_by_range(T_300_C1_1,temperature_ranges_all)
indices_temp_300_150_C1_2 = get_indices_by_range(T_300_C1_2,temperature_ranges_all)
indices_temp_300_150_C1_3 = get_indices_by_range(T_300_C1_3,temperature_ranges_all)
indices_temp_300_150_C1_4 = get_indices_by_range(T_300_C1_4,temperature_ranges_all)
# Lista de listas de índices por archivo de temperatura
indices_by_temp = [indices_temp_300_150_C1_2, indices_temp_300_150_C1_3, indices_temp_300_150_C1_4]

Temp_300_C1 = []
Temp_300_C1_err = []
tau_300_C1 = []
tau_300_C1_err = []
# Cálculo de promedios y desviaciones estándar por rango de temperatura
for i in range(len(temperature_ranges_all)):
    # Promedio para T_300
    Temp_300_C1.append(np.mean(np.concatenate([T_300_C1_2[indices_temp_300_150_C1_2[i]],
                                            T_300_C1_3[indices_temp_300_150_C1_3[i]],
                                            T_300_C1_4[indices_temp_300_150_C1_4[i]]])))

    Temp_300_C1_err.append(np.std(np.concatenate([T_300_C1_2[indices_temp_300_150_C1_2[i]],
                                                T_300_C1_3[indices_temp_300_150_C1_3[i]],
                                                T_300_C1_4[indices_temp_300_150_C1_4[i]]]))),

    tau_300_C1.append(np.mean(np.concatenate([tau_300_C1_2[indices_temp_300_150_C1_2[i]],
                                           tau_300_C1_3[indices_temp_300_150_C1_3[i]],
                                           tau_300_C1_4[indices_temp_300_150_C1_4[i]]])))

    tau_300_C1_err.append(np.std(np.concatenate([tau_300_C1_2[indices_temp_300_150_C1_2[i]],
                                                 tau_300_C1_3[indices_temp_300_150_C1_3[i]],
                                            tau_300_C1_4[indices_temp_300_150_C1_4[i]]])))

#remuevo elementos nan
Temp_300_C1 = [i for i in Temp_300_C1 if ~np.isnan(i)]
Temp_300_C1_err = [i for i in Temp_300_C1_err if ~np.isnan(i)]
tau_300_C1 = [i for i in tau_300_C1 if ~np.isnan(i)]
tau_300_C1_err = [i for i in tau_300_C1_err if ~np.isnan(i)]

indices_temp_300_150_C2_1 = get_indices_by_range(T_300_C2_1,temperature_ranges_all)
indices_temp_300_150_C2_2 = get_indices_by_range(T_300_C2_2,temperature_ranges_all)
indices_temp_300_150_C2_3 = get_indices_by_range(T_300_C2_3,temperature_ranges_all)
indices_temp_300_150_C2_4 = get_indices_by_range(T_300_C2_4,temperature_ranges_all)
#### C2
Temp_300_C2 = []
Temp_300_C2_err = []
tau_300_C2 = []
tau_300_C2_err = []
for i in range(len(temperature_ranges_all)):
    # Promedio para T_300
    Temp_300_C2.append(np.mean(np.concatenate([T_300_C2_1[indices_temp_300_150_C2_1[i]],
                                               T_300_C2_2[indices_temp_300_150_C2_2[i]],
                                            T_300_C2_3[indices_temp_300_150_C2_3[i]],
                                            T_300_C2_4[indices_temp_300_150_C2_4[i]]])))

    Temp_300_C2_err.append(np.std(np.concatenate([T_300_C2_1[indices_temp_300_150_C2_1[i]],
                                                  T_300_C2_2[indices_temp_300_150_C2_2[i]],
                                                T_300_C2_3[indices_temp_300_150_C2_3[i]],
                                                T_300_C2_4[indices_temp_300_150_C2_4[i]]]))),

    tau_300_C2.append(np.mean(np.concatenate([tau_300_C2_1[indices_temp_300_150_C2_1[i]],
                                              tau_300_C2_2[indices_temp_300_150_C2_2[i]],
                                           tau_300_C2_3[indices_temp_300_150_C2_3[i]],
                                           tau_300_C2_4[indices_temp_300_150_C2_4[i]]])))

    tau_300_C2_err.append(np.std(np.concatenate([tau_300_C2_1[indices_temp_300_150_C2_1[i]],
                                                tau_300_C2_2[indices_temp_300_150_C2_2[i]],
                                            tau_300_C2_3[indices_temp_300_150_C2_3[i]],
                                            tau_300_C2_4[indices_temp_300_150_C2_4[i]]])))

#remuevo elementos nan
Temp_300_C2 = [i for i in Temp_300_C2 if ~np.isnan(i)]
Temp_300_C2_err = [i for i in Temp_300_C2_err if ~np.isnan(i)]
tau_300_C2 = [i for i in tau_300_C2 if ~np.isnan(i)]
tau_300_C2_err = [i for i in tau_300_C2_err if ~np.isnan(i)]


#%%

fig,(ax,ax2)=plt.subplots(nrows=2,figsize=(11,7),constrained_layout=True,sharex=True)
ax.set_title('All',loc='left')
ax.plot(T_300_C1_1,tau_300_C1_1,'-',label='C1')
ax.plot(T_300_C1_2,tau_300_C1_2,'-',label='C1')
ax.plot(T_300_C1_3,tau_300_C1_3,'-',label='C1')
ax.plot(T_300_C1_4,tau_300_C1_4,'-',label='C1')
ax.plot(T_300_C2_1,tau_300_C2_1,'.-',lw=0.7,label='C2')
ax.plot(T_300_C2_2,tau_300_C2_2,'.-',lw=0.7,label='C2')
ax.plot(T_300_C2_3,tau_300_C2_3,'.-',lw=0.7,label='C2')
ax.plot(T_300_C2_4,tau_300_C2_4,'.-',lw=0.7,label='C2')

ax2.set_title('Promedios',loc='left')
ax2.errorbar(x=Temp_300_C1,y=tau_300_C1,xerr=Temp_300_C1_err,yerr=tau_300_C1_err,fmt='.-',capsize=3,label='C1')
ax2.errorbar(x=Temp_300_C2,y=tau_300_C2,xerr=Temp_300_C2_err,yerr=tau_300_C2_err,fmt='.-',capsize=3,label='C2')
ax2.set_xlabel('T (°C)')


for a in [ax,ax2]:
    a.grid()
    a.set_ylabel(r'$\tau$ (s)')
    a.legend(title=f'''C1 = {meta_C1_2["Concentracion g/m^3"]/1e3:.1f} g/L    C2 = {meta_C2_2["Concentracion g/m^3"]/1e3:.1f} g/L''',ncol=2)
plt.suptitle(f'$\\tau$ vs T\nNE5X en SV\n$f$ = 300 kHz    $H_0$ = 57 kA/m',fontsize=13)
plt.savefig('C1_C2_comparacion.png',dpi=300)
#%% Busco las de 300 57 en NE5X para comparar 
resultados_C0_300 = glob(os.path.join('./C0', '*resultados*.txt'))
resultados_C0_300.sort()
meta_C0_1,files_1,time_1,T_300_C0_1,Mr_300_C0_1,Hc_300_C0_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_0_1,dphi_fem_0_1,SAR_300_C0_1,tau_300_C0_1,N1 = lector_resultados(resultados_C0_300[0])
meta_C0_2,files_2,time_2,T_300_C0_2,Mr_300_C0_2,Hc_300_C0_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_0_2,dphi_fem_0_2,SAR_300_C0_2,tau_300_C0_2,N2 = lector_resultados(resultados_C0_300[1])
meta_C0_3,files_3,time_3,T_300_C0_3,Mr_300_C0_3,Hc_300_C0_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_0_3,dphi_fem_0_3,SAR_300_C0_3,tau_300_C0_3,N3 = lector_resultados(resultados_C0_300[2])
meta_C0_4,files_4,time_4,T_300_C0_4,Mr_300_C0_4,Hc_300_C0_4,campo_max_4,mag_max_4,xi_M_0_4,frecuencia_fund_4,magnitud_fund_0_4,dphi_fem_0_4,SAR_300_C0_4,tau_300_C0_4,N3 = lector_resultados(resultados_C0_300[3])

#% Tau vs Temp
fig,((ax,ax2),(ax3,ax4))= plt.subplots(nrows=2,ncols=2,figsize=(12,8),constrained_layout=True,sharex=True)

ax.plot(T_300_C0_1,tau_300_C0_1,'.-',label='1')
ax.plot(T_300_C0_2,tau_300_C0_2,'.-',label='2')
ax.plot(T_300_C0_3,tau_300_C0_3,'.-',label='3')
ax.plot(T_300_C0_4,tau_300_C0_4,'.-',label='4')
ax.set_title(r'$\tau$')
ax.set_ylabel(r'$\tau$ (s)')

ax2.plot(T_300_C0_1,SAR_300_C0_1,'.-',label='1')
ax2.plot(T_300_C0_2,SAR_300_C0_2,'.-',label='2')
ax2.plot(T_300_C0_3,SAR_300_C0_3,'.-',label='3')
ax2.plot(T_300_C0_4,SAR_300_C0_4,'.-',label='4')
ax2.set_title('SAR')
ax2.set_ylabel('SAR (W/g)')

ax3.plot(T_300_C0_1,Mr_300_C0_1,'.-',label='1')
ax3.plot(T_300_C0_2,Mr_300_C0_2,'.-',label='2')
ax3.plot(T_300_C0_3,Mr_300_C0_3,'.-',label='3')
ax3.plot(T_300_C0_4,Mr_300_C0_4,'.-',label='4')
ax3.set_title('M$_R$')
ax3.set_ylabel('M$_R$ (A/m)')

ax4.plot(T_300_C0_1,Hc_300_C0_1,'.-',label='1')
ax4.plot(T_300_C0_2,Hc_300_C0_2,'.-',label='2')
ax4.plot(T_300_C0_3,Hc_300_C0_3,'.-',label='3')
ax4.plot(T_300_C0_4,Hc_300_C0_4,'.-',label='4')
ax4.set_title('H$_C$')
ax4.set_ylabel('H$_C$  (kA/m)')

for a in [ax,ax2,ax3,ax4]:
    a.legend()
    a.grid()
    
ax3.set_xlabel('T (°C)')
ax4.set_xlabel('T (°C)')
plt.suptitle('Comparativa C0 - NE5X - 300 kHz 57 kA/m')
plt.savefig('C0_300_57_NE5X_comparativa.png',dpi=300)
plt.show()
#%% promedio C0
indices_temp_300_150_C0_2 = get_indices_by_range(T_300_C0_2,temperature_ranges_all)
indices_temp_300_150_C0_3 = get_indices_by_range(T_300_C0_3,temperature_ranges_all)
indices_temp_300_150_C0_4 = get_indices_by_range(T_300_C0_4,temperature_ranges_all)
# Lista de listas de índices por archivo de temperatura
indices_by_temp = [indices_temp_300_150_C0_2, indices_temp_300_150_C0_3, indices_temp_300_150_C0_4]

Temp_300_C0 = []
Temp_300_C0_err = []
tau_300_C0 = []
tau_300_C0_err = []
# Cálculo de promedios y desviaciones estándar por rango de temperatura
for i in range(len(temperature_ranges_all)):
    # Promedio para T_300
    Temp_300_C0.append(np.mean(np.concatenate([T_300_C0_2[indices_temp_300_150_C0_2[i]],
                                            T_300_C0_3[indices_temp_300_150_C0_3[i]],
                                            T_300_C0_4[indices_temp_300_150_C0_4[i]]])))

    Temp_300_C0_err.append(np.std(np.concatenate([T_300_C0_2[indices_temp_300_150_C0_2[i]],
                                                T_300_C0_3[indices_temp_300_150_C0_3[i]],
                                                T_300_C0_4[indices_temp_300_150_C0_4[i]]]))),

    tau_300_C0.append(np.mean(np.concatenate([tau_300_C0_2[indices_temp_300_150_C0_2[i]],
                                           tau_300_C0_3[indices_temp_300_150_C0_3[i]],
                                           tau_300_C0_4[indices_temp_300_150_C0_4[i]]])))

    tau_300_C0_err.append(np.std(np.concatenate([tau_300_C0_2[indices_temp_300_150_C0_2[i]],
                                            tau_300_C0_3[indices_temp_300_150_C0_3[i]],
                                            tau_300_C0_4[indices_temp_300_150_C0_4[i]]])))
Temp_300_C0 = [i for i in Temp_300_C0 if ~np.isnan(i)]
Temp_300_C0_err = [i for i in Temp_300_C0_err if ~np.isnan(i)]
tau_300_C0 = [i for i in tau_300_C0 if ~np.isnan(i)]
tau_300_C0_err = [i for i in tau_300_C0_err if ~np.isnan(i)]

fig,ax=plt.subplots(nrows=1,figsize=(12,5),constrained_layout=True)
ax.errorbar(x=Temp_300_C0,y=tau_300_C0,xerr=Temp_300_C0_err,yerr=tau_300_C0_err,fmt='.-',capsize=3,label='C0 (aq)')
ax.errorbar(x=Temp_300_C1,y=tau_300_C1,xerr=Temp_300_C1_err,yerr=tau_300_C1_err,fmt='.-',capsize=3,label='C1 (SV)')
ax.errorbar(x=Temp_300_C2,y=tau_300_C2,xerr=Temp_300_C2_err,yerr=tau_300_C2_err,fmt='.-',capsize=3,label='C2 (SV)')

ax.legend(title=f'C0 = {meta_C0_1["Concentracion g/m^3"]/1e3:.1f} g/L    C1 = {meta_C1_2["Concentracion g/m^3"]/1e3:.1f} g/L    C2 = {meta_C2_2["Concentracion g/m^3"]/1e3:.1f} g/L',ncol=3)
ax.grid()
ax.set_xlabel('T (°C)')
ax.set_ylabel(r'$\tau$ (s)')
ax.set_title(f'$\\tau$ vs T\nNE5X\n$f$ = 300 kHz      $H_0$ = 57 kA/m')

# %%
