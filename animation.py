import xarray as xr
import pandas as pd
import numpy as np
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/snap/bin/ffmpeg'
#plt.style.use('seaborn-whitegrid')
import matplotlib.gridspec as gridspec
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os


mpl.rcParams['font.size'] = 10
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

cmap = cm.get_cmap('Blues', 256)
newcolors = cmap(np.linspace(0, 1, 256))
black = np.array([0/256, 0/256, 0/256, 1])
newcolors[:1, :] = black
newcmp = ListedColormap(newcolors)


def read_csv(file):
    df = pd.read_csv(file)
    df = df.set_index(pd.DatetimeIndex(df['date_time']))
    return df


def open_data(file,variable):
    ds = xr.open_dataset(file)
    ds = ds[variable]
    return ds

def pull_pixel(ds,station):
    pixel = ds.sel(x=station[0], y=station[1])
    return pixel



def sort_data(ds,stations):
    pixel_dict = {}
    for station in station_dict:
        pixel_dict[station] = pull_pixel(ds, station_dict[station])

    return ds, pixel_dict



def plot_images(i,ax,ds,vmin,vmax):
    ax.set_title(str(ds[i].time.values))
    im = ax.imshow(ds[i],extent=[ds[i].x.values[0],ds[i].x.values[-1],ds[i].y.values[-1],ds[i].y.values[0]],cmap=newcmp,vmin=vmin,vmax=vmax)
    return im

def plot_station_coords(ax,station,color,s,marker):
    ax.scatter(station[0],station[1],facecolors=color, edgecolors='k',s=s,marker=marker)

def plot_stations(i,ax,station,color,ls):
    ax.plot(station[0:i].time.values,station[0:i].values,c=color,linestyle=ls,zorder=1)

def plot_between(i,ax,station,color,ls):
    ax.fill_between(station[0:i].time.values,station[0:i].values,color=color,alpha=0.3,zorder=0)

def plot_pits(ax,x,y,color,s):
    ax.scatter(x,y,zorder=10,facecolors=color, edgecolors='k',s=s)


def clear_ax(ax):
    ax.clear()



wy_dict = {'2008':'8064','2009':'8040','2010':'8040','2011':'8040',
           '2012':'8064','2013':'8040','2014':'8040','2015':'8040',
           '2016':'8064','2017':'8040'}

station_dict = {'sasp':[261630.0,4198955.0],'sbsp':[260320.0,4198985.0]}

wy = '2012'

writer = FFMpegWriter(fps=6)

fig = plt.figure(figsize=(12,10))
gs = gridspec.GridSpec(nrows=5, ncols=4, height_ratios=[1,1,0.1,1,1],width_ratios=[1,1,1,1],wspace=0.6,hspace=0.2)
ax1 = fig.add_subplot(gs[0:2,0:2])
ax2 = fig.add_subplot(gs[0:2,2:4])
axcb = fig.add_subplot(gs[2,:])
ax3 = fig.add_subplot(gs[3,:])
ax3b = ax3.twinx()
ax4 = fig.add_subplot(gs[4,:])
ax4b = ax4.twinx()

def animate(i):
    print(i)
    clear_ax(ax1)
    clear_ax(ax2)
    clear_ax(axcb)
    clear_ax(ax3)
    clear_ax(ax3b)
    clear_ax(ax4)
    clear_ax(ax4b)

    dir = '/media/jon/J/model/output/sbbsa/devel/wy' + wy + '/test2/data/data0000_' + wy_dict[wy] + '/smrfOutputs'
    os.chdir(dir)
    ds = open_data('net_solar_original.nc', 'net_solar')
    ds,pixel_dict = sort_data(ds, station_dict)
    for station in pixel_dict:
        pixel_dict[station]=pixel_dict[station].resample(time='24H',base=11 + 0).sum()
    plot_between(i, ax3b, pixel_dict['sasp'],'k',ls='-')
    plot_between(i, ax4b, pixel_dict['sbsp'],'k',ls='-')



    dir = '/media/jon/J/model/output/sbbsa/devel/wy' + wy + '/test2/data/data0000_' + wy_dict[wy] + '/smrfOutputs'
    os.chdir(dir)
    ds = open_data('net_solar.nc', 'net_solar')
    ds,pixel_dict = sort_data(ds, station_dict)
    for station in pixel_dict:
        pixel_dict[station]=pixel_dict[station].resample(time='24H',base=11 + 0).sum()
    plot_between(i, ax3b, pixel_dict['sasp'],'salmon',ls='-')
    plot_between(i, ax4b, pixel_dict['sbsp'],'salmon',ls='-')


    dir = '/media/jon/J/model/output/sbbsa/devel/wy' + wy + '/test2/runs/run0000_' + wy_dict[wy]
    os.chdir(dir)
    ds = open_data('snow.nc', 'thickness')
    ds,pixel_dict = sort_data(ds, station_dict)
    im = plot_images(i, ax2, ds,vmin=0,vmax=4)
    plot_station_coords(ax2, station_dict['sasp'],'white',s=40,marker='$A$')
    plot_station_coords(ax2, station_dict['sbsp'],'white',s=40,marker='$B$')
    plot_stations(i, ax3, pixel_dict['sasp'],'red',ls='-')
    plot_stations(i, ax4, pixel_dict['sbsp'],'red',ls='-')

    dir = '/media/jon/J/model/output/sbbsa/devel/wy' + wy + '/test2/runs/run0000_' + wy_dict[wy]
    os.chdir(dir)
    ds = open_data('snow_original.nc', 'thickness')
    ds,pixel_dict = sort_data(ds,station_dict)
    im = plot_images(i, ax1, ds,vmin=0,vmax=4)
    plot_station_coords(ax1, station_dict['sasp'],'white',s=40,marker='$A$')
    plot_station_coords(ax1, station_dict['sbsp'],'white',s=40,marker='$B$')
    plot_stations(i, ax3, pixel_dict['sasp'],'black',ls='-')
    plot_stations(i, ax4, pixel_dict['sbsp'],'black',ls='-')




    dir = '/home/jon/projects/senator_beck/data/pit_data/'
    os.chdir(dir)
    df = read_csv('SASP_pit_measurements.csv')
    plot_pits(ax3, x=df.index, y=df.depth, color='k',s=40)


    dir = '/home/jon/projects/senator_beck/data/pit_data/'
    os.chdir(dir)
    df = read_csv('SBSP_pit_measurements.csv')
    plot_pits(ax4, x=df.index, y=df.depth, color='k',s=40)

    os.chdir('/home/jon/projects/thesis/data/')
    df = pd.read_csv('dust_events.csv')
    df = df.set_index(pd.date_range(start='10/1/'+str(int(wy)-1), periods=366, freq='D'))
    df = df[wy]
    df = df.dropna()
    for event in df.index:
        ax3b.axvline(event,linewidth=2, color='gray',zorder=0,linestyle='--',alpha=0.6)




    cb = colorbar(im, cax=axcb, orientation="horizontal")
    axcb.xaxis.set_ticks_position("top")
    axcb.set_xlabel('Snow Depth (m)')



    ax1.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    ax1.grid()
    ax1.set_xlabel('Clean Snow')

    ax2.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    ax2.grid()
    ax2.set_xlabel('In Situ Albedo')

    ax3.grid()
    ax3.set_xlim(ds.time.values[0],ds.time.values[-1])
    ax3.set_ylim(0,4)
    ax3.set_ylabel('Snow Depth (m)')
    ax3.set_zorder(1)
    ax3.patch.set_visible(False)
    ax3.text(ds.time.values[10],3.5,'A',color='black')
    ax3b.set_ylabel(r'Net Solar ($Wm^{-2}$)')
    ax3b.set_ylim(0,8000)
    ax3b.set_zorder(0)


    ax4.grid()
    ax4.set_xlim(ds.time.values[0],ds.time.values[-1])
    ax4.set_ylim(0,4)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Snow Depth (m)')
    ax4.set_zorder(1)
    ax4.text(ds.time.values[10],3.5,'B',color='black')
    ax4.patch.set_visible(False)
    ax4b.set_ylim(0,8000)
    ax4b.set_ylabel(r'Net Solar ($Wm^{-2}$)')
    ax4b.set_zorder(0)

    gc.collect()





    #plt.tight_layout()






os.chdir('/home/jon/projects/thesis/figures/animations')
with writer.saving(fig,wy+"t.mp4",dpi=100):
    for i in range(150,330):
        animate(i)
        writer.grab_frame()


# anim = animation.FuncAnimation(fig, animate, interval=1,frames=5)
#animate(300)

# os.chdir('/home/jon/projects/thesis/figures/animations')
# anim.save('test.mp4',writer=writer)
#plt.savefig('test.png')
# plt.show()
