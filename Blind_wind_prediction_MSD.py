
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import root_mean_squared_error,r2_score
import seaborn as sns

names=[
'arolla',
'langenferner',
'schiaparelli',
'Trakarding',
'yala',
'lirung',
'satopanth',
'wm2',
'pm3',
'pm2',
'pm1',
'djankuat',
'chotashigri',
'wm1',
'drangdrung',
'pyramid',
'pheriche',
'bishop',
'weisssee',
'hef',
'Rotmoos',
'lukla',
'phortse',
'namche',
'balcony',
'camp2',
'pm4',
'southcol',
'Denalipass',
'kalapathar'
]
# Leave one out cross validation

rmse_lres=[]
rmse_lreg=[]
rmse_era5=[]

yodata=[]
ypdata=[]
ypldata=[]
yedata=[]

params=pd.read_csv(r'c:\Users\krish\Documents\August_draft\August_draft\Paper_sankoo\Nepal_stations\New_bigtable\NEWbigtableparams_backup_dummy_newsensitivity.csv')
params=params.iloc[0:15]
glacier_stations=params.iloc[:]
glacier_stations['rel_z']=(glacier_stations['station_z']-glacier_stations['Z_min'])/(glacier_stations['Z_max']-glacier_stations['Z_min'])
glacier_stations['rel_distancefromterminus']=glacier_stations['Distance from terminus']/glacier_stations['glacier_length']
glacier_stations['rel_distancefromhead']=glacier_stations['Distance from head']/glacier_stations['glacier_length']
glacier_stations['rel_distancefromcenterline']=glacier_stations['Distance from centerline']/glacier_stations['glacier_width_at_station']
glacier_stations['rel_distancefromedge']=1-(glacier_stations['Distance from centerline']/glacier_stations['glacier_width_at_station'])
glacier_stations['local_aspect_normalised']=np.cos(np.degrees(glacier_stations['aspect_1km_50p']))
glacier_stations['asrgi_aspect_normalised']=np.cos(np.degrees(glacier_stations['rgi_aspect']))
params=glacier_stations

x_data_mean_wind=params[['glacier_width_at_station', 'Distance from centerline', 'continentality', 'mean_wind_era5', 'T_range_continentality']]
y_data_mean_wind=params['mean_wind_obs']
x_data_sensitivity=params[['glacier_length', 'glacier_width_at_station', 'Distance from terminus', 'rel_z']]
y_data_sensitivity=params['sensitivity']
y_data_linsensitivity=params['sensitivity']
x_data_delay=params[['Distance from centerline', 'rel_distancefromterminus']]
y_data_delay=params['delay']


for i in range(15):
    if i==1:
        continue
    # mean wind prediction
    from sklearn.linear_model import LinearRegression
    Lr=HuberRegressor()
    leave=[i,1]
    model=Lr.fit(X=x_data_mean_wind.drop(index=leave),y=y_data_mean_wind.drop(index=leave))
    x_validate_mean_wind=x_data_mean_wind.iloc[i:i+1]
    Mean_wind=model.predict(x_validate_mean_wind)


    # sensitivity prediction
    from sklearn.linear_model import LinearRegression
    Lr=HuberRegressor()
    leave=[i,1]
    model=Lr.fit(X=x_data_sensitivity.drop(index=leave),y=y_data_sensitivity.drop(index=leave))
    x_validate_sensitivity=x_data_sensitivity.iloc[i:i+1]
    S=model.predict(x_validate_sensitivity)

    # linear sensitivity prediction
    from sklearn.linear_model import LinearRegression
    Lr=HuberRegressor()
    leave=[i,1]
    model=Lr.fit(X=x_data_sensitivity.drop(index=leave),y=y_data_linsensitivity.drop(index=leave))
    x_validate_sensitivity=x_data_sensitivity.iloc[i:i+1]
    S_reg=model.predict(x_validate_sensitivity)

    # delay prediction
    from sklearn.linear_model import LinearRegression
    Lr=HuberRegressor()
    leave=[i,1]
    model=Lr.fit(X=x_data_delay.drop(index=leave),y=y_data_delay.drop(index=leave))
    x_validate_delay=x_data_delay.iloc[i:i+1]
    D=model.predict(x_validate_delay)   


    A=S
    B=-D*S


    D_reg=0
    A_reg=S_reg
    B_reg=0

    # Time series
    name=names[i]
    station_data=pd.read_csv('bigdata_0.1_criteria//'+name+'.csv')
    station_data['datetime']=pd.to_datetime(station_data['datetime'])
    linear_model=station_data.copy()
    linear_model['temp_model']=linear_model['temp_model']-linear_model['temp_model'].mean()
    linear_model['ws']=linear_model['ws']-linear_model['ws'].mean()
    linear_model['ws_model']=linear_model['ws_model']-linear_model['ws_model'].mean()

    lres_wind_speed=np.array(((A*linear_model['temp_model']+B*linear_model['temp_model'].diff())).replace(np.nan,0))+Mean_wind
    lreg_wind_speed=np.array(((A_reg*linear_model['temp_model']+B_reg*linear_model['temp_model'].diff())).replace(np.nan,0))+Mean_wind
    observed_wind_speed=station_data['ws'].values
    era5_wind_speed=station_data['ws_model'].values

    # if not (i==3):
    for l in range(23):
        yodata.append(observed_wind_speed[l])
        ypdata.append(lres_wind_speed[l])
        ypldata.append(lreg_wind_speed[l])
        yedata.append(era5_wind_speed[l])
    
jaja=pd.DataFrame()

jaja['u_era5']=yedata#-jaja['u_obs']
jaja['u_S']=ypldata#-jaja['u_obs']
jaja['u_SD']=ypdata
jaja['u_obs']=yodata
jaja['glacier_number']=np.arange(322)//23



plt.figure(figsize=(9,8))
plt.rcParams['font.size']=20
plt.grid(True,zorder=-1)
# plt.axhline(0,color='black',linewidth=1,linestyle='--')
plt.axline((0,0),(1,1),color='black',linewidth=2,linestyle='-')
plt.vlines(jaja['u_obs'],jaja['u_era5'],jaja['u_SD'],linewidth=0.4,linestyle='--',color='gray',zorder=1)
# sns.regplot(data=jaja, x='u_obs',y='their_res',ci=99)
import matplotlib
cmap=matplotlib.colormaps['prism']
plt.scatter(jaja['u_obs'],jaja['u_SD'],label="RMSE1=0.55 "+r"$\text{ms}{}^{-1})$",marker='^',edgecolor="black",s= 100,linewidth=0.5,c='deeppink',zorder=100)
plt.scatter(jaja['u_obs'],jaja['u_era5'],label="RMSE2=1.89 "+r"$\text{ms}{}^{-1})$",edgecolor="black",s=100,linewidth=0.5,c='lightskyblue',zorder=2)
# plt.axvline()
plt.xlabel(r"$u_{obs}\:(\text{ms}{}^{-1})$")
plt.ylabel(r"$u_{pred}\:(\text{ms}{}^{-1})$")
plt.legend(loc='upper left')

# plt.xlim(-2,2)
# plt.ylim(-2,2)
# jaja.dropna(inplace=True)

ax = plt.gca()
ax.set_facecolor('white')  # Light gray background color
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
# Show the plot
plt.xlim(0,6)
plt.ylim(0,6)
plt.minorticks_on()
plt.tight_layout()
plt.savefig("Blind_era5_linres_hourly.pdf")
plt.show()
# plt.close()


###########################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Blind era5 lin regression
###########################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# plt.figure(figsize=(9,8))
# plt.rcParams['font.size']=15
# plt.grid(True,zorder=-1)
# jaja['their_res']=jaja['our_linres']
# jaja['our_linres']=jaja['our_res']
# # plt.axhline(0,color='black',linewidth=1,linestyle='--')
# plt.axline((0,0),(1,1),color='black',linewidth=2,linestyle='-')
# plt.vlines(jaja['u_obs'],jaja['our_linres'],jaja['their_res'],linewidth=0.8,linestyle='--',color='navy',zorder=1)
# plt.scatter(jaja['u_obs'],jaja['their_res'],label='ERA_5+lin_regression, n=15',edgecolor="black",s=50,linewidth=0.2,c='navy',zorder=2)
# # sns.regplot(data=jaja, x='u_obs',y='their_res',ci=99)
# import matplotlib
# cmap=matplotlib.colormaps['prism']
# plt.scatter(jaja['u_obs'],jaja['our_linres'],label="ERA_5+lin_response, n=15",edgecolor='black',linewidth=0.3,s=30,c='red',marker='^',zorder=2,cmap='Paired')
# # plt.axvline()
# plt.xlabel(r"$wind_{obs}\:(m/s)$")
# plt.ylabel(r"$wind_{model}\:(m/s)$")
# plt.legend(loc='upper left')

# # plt.xlim(-2,2)
# # plt.ylim(-2,2)
# # jaja.dropna(inplace=True)
# plt.text(1,3.5,"RMSE="+str(round(root_mean_squared_error(jaja['u_obs'],jaja['their_res']),3))+"\nR^2_adj="+str(round(r2_score(jaja['u_obs'],jaja['their_res']),3)),color='navy',fontdict={'size':15})
# plt.text(1,3,"RMSE="+str(round(root_mean_squared_error(jaja['u_obs'],jaja['our_linres']),3))+"\nR^2_adj="+str(round(r2_score(jaja['u_obs'],jaja['our_linres']),3)),color='red',fontdict={'size':15})

# ax = plt.gca()
# ax.set_facecolor('white')  # Light gray background color
# plt.xlim(0,5)
# plt.ylim(0,5)
# # Show the plot
# plt.show()
# plt.tight_layout()
# plt.savefig("Blind_era5_linregr.pdf")
# # plt.close()