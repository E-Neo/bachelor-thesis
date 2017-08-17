import io
import os
import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats


def read_data(filename):
    with open(filename, 'r') as f:
        lines = [line.replace('NAN', 'nan') for line in f]
        sio = io.StringIO(''.join(lines[1:2] + lines[4:]))
        df = pd.read_csv(sio, dtype=float, parse_dates=["TIMESTAMP"])
        df.set_index('TIMESTAMP', inplace=True)
        return df


def read_CR5000(path):
    subdir = sorted(os.listdir(path))
    dfs = [read_data(path + "/{}/CR5000_flux.dat".format(s))
           for s in subdir]
    df = pd.concat(dfs)
    return df


def get_radiation(start_date, end_date, freq='30min'):
    hefei = pvlib.location.Location(31.863890, 117.280830, 'Asia/Shanghai')
    times = pd.DatetimeIndex(start=start_date, end=end_date,
                             freq=freq, tz=hefei.tz)
    cs = hefei.get_clearsky(times)
    cs.tz_localize(None, copy=False)
    cs.index.name = 'TIMESTAMP'
    return cs


def set_dS(df):
    if 'Rs_downwell_Avg' in df:
        df['Rn'] = df['Rs_downwell_Avg'] + df['Rl_downwell_Avg']\
                   - df['Rs_upwell_Avg'] - df['Rl_upwell_Avg']
    else:
        df['Rn'] = df['DR_Avg'] + df['DLR_Avg']\
                   - df['UR_Avg'] - df['ULR_Avg']
    df['dS'] = df['Rn'] - df['Hs'] - df['LE_irga']


def set_albedo(df):
    if 'Rs_downwell_Avg' in df:
        df['albedo'] = df['Rs_upwell_Avg'] / df['Rs_downwell_Avg']
        df.loc[df['Rs_downwell_Avg'] < 10, 'albedo'] = 0
    else:
        df['albedo'] = df['UR_Avg'] / df['DR_Avg']
        df.loc[df['DR_Avg'] < 10, 'albedo'] = 0


def get_raw():
    df1_path = '/home/e-neo/Workspace/homework/weather/data01/CR3000_flux.dat'
    df2_path = '/home/e-neo/Workspace/homework/weather/CR5000'
    df1, df2 = read_data(df1_path), read_CR5000(df2_path)
    df2.sort_index(inplace=True)
    df2.drop_duplicates(inplace=True)
    # df1 = pd.concat([df1['2014-11-10':'2015-04-04'],
    #                  df1['2015-10-21':'2016-06-01']]).asfreq('30T')
    # df2 = pd.concat([df2['2014-11-10':'2015-04-04'],
    #                  df2['2015-10-21':'2016-06-01']]).asfreq('10T')
    df1 = df1['2015-10-21':'2016-06-01'].asfreq('30T')
    df2 = df2['2015-10-21':'2016-06-01'].asfreq('10T')
    return df1, df2.resample('30T').mean()


def clean_Hs(df):
    df.loc[(df['Hs'] < -50) |
           (df['Hs'] > 300), 'Hs'] = np.nan
    return df


def clean_LE(df, n):
    df.loc[(df['LE_irga'] > 500) |
           (df['LE_irga'] < -200), 'LE_irga'] = np.nan
    for i in range(n):
        sigma = df['LE_irga'].diff().std()
        mask = df['LE_irga'].diff().abs() > 3*sigma
        df.loc[mask, 'LE_irga'] = np.nan
    return df


def clean_CO2(df, n):
    CO2 = 'CO2_mean' if 'CO2_mean' in df else 'co2_Avg'
    df.loc[(df[CO2] > 1000) | (df[CO2] < 0), CO2] = np.nan
    for i in range(n):
        sigma = df[CO2].diff().std()
        mask = df[CO2].diff().abs() > 3*sigma
        df.loc[mask, CO2] = np.nan
    return df


def clean_H2O(df, n):
    H2O = 'H2O_mean' if 'H2O_mean' in df else 'h2o_Avg'
    df.loc[(df[H2O] > 25) | (df[H2O] < -5), H2O] = np.nan
    for i in range(n):
        sigma = df[H2O].diff().std()
        mask = df[H2O].diff().abs() > 3*sigma
        df.loc[mask, H2O] = np.nan
    return df


def clean_wnd_spd(df):
    df.loc[df['wnd_spd'] > 8, 'wnd_spd'] = np.nan
    return df


def clean_Uz(df):
    df.loc[(df['Uz_Avg'] > 0.4) | (df['Uz_Avg'] < -0.5), 'Uz_Avg'] = np.nan
    return df


def clean_dS(df):
    df.loc[(df['dS'] > 750) | (df['dS'] < -250)] = np.nan
    return df


def get_clean():
    df1, df2 = get_raw()
    df1, df2 = clean_Hs(df1), clean_Hs(df2)
    df1, df2 = clean_LE(df1, 1), clean_LE(df2, 1)
    df1, df2 = clean_H2O(df1, 1), clean_H2O(df2, 1)
    df1, df2 = clean_CO2(df1, 2), clean_CO2(df2, 2)
    df1, df2 = clean_wnd_spd(df1), clean_wnd_spd(df2)
    df1, df2 = clean_Uz(df1), clean_Uz(df2)
    df1.index = df1.index - pd.Timedelta(minutes=30)
    df2['DR_Avg'] = df2['DR_Avg']*0.86
    df2['UR_Avg'] = df2['UR_Avg']*0.86

    set_dS(df1)
    set_dS(df2)
    set_albedo(df1)
    set_albedo(df2)
    df1, df2 = clean_dS(df1), clean_dS(df2)

    # cs = pd.concat([get_radiation('2014-11-10', '2015-04-04'),
    #                 get_radiation('2015-10-21', '2016-06-02')]).asfreq('30T')
    cs = get_radiation('2015-10-21', '2016-06-02').asfreq('30T')
    return df1, df2, cs


def new_get_clean():
    # cs = pd.concat([get_radiation('2014-11-10', '2015-04-04'),
    #                 get_radiation('2015-10-21', '2016-06-02')]).asfreq('30T')
    cs = get_radiation('2015-10-21', '2016-06-02').asfreq('30T')

    df1, df2 = get_raw()
    df1, df2 = clean_Hs(df1), clean_Hs(df2)
    df1, df2 = clean_LE(df1, 1), clean_LE(df2, 1)
    df1, df2 = clean_H2O(df1, 1), clean_H2O(df2, 1)
    df1, df2 = clean_CO2(df1, 2), clean_CO2(df2, 2)
    df1, df2 = clean_wnd_spd(df1), clean_wnd_spd(df2)
    df1, df2 = clean_Uz(df1), clean_Uz(df2)
    df1.index = df1.index - pd.Timedelta(minutes=30)
    # df2['DR_Avg'] = df2['DR_Avg']*0.86
    # df2['UR_Avg'] = df2['UR_Avg']*0.86
    df2.loc[df2['DR_Avg'] / cs['ghi'] > 0.7,
            'DR_Avg'] = 0.86 * df2.loc[df2['DR_Avg'] / cs['ghi'] > 0.7,
                                       'DR_Avg']
    set_dS(df1)
    set_dS(df2)
    set_albedo(df1)
    set_albedo(df2)
    df1, df2 = clean_dS(df1), clean_dS(df2)
    return df1, df2, cs


def select_date(df1, df2, cs, start_date, end_date):
    ndf1 = df1[start_date:end_date]
    ndf2 = df2[start_date:end_date]
    ncs = cs[start_date:end_date]
    return ndf1, ndf2, ncs


def get_daily(df1, df2, cs):
    ddf1 = df1.resample('D').sum()
    ddf2 = df2.resample('D').sum()
    dcs = cs.resample('D').sum()
    return ddf1, ddf2, dcs


def select_sunny(df1, df2, cs):
    ddf1, ddf2, dcs = get_daily(df1, df2, cs)
    t1 = ddf1['Rs_downwell_Avg'] / dcs['ghi']
    t2 = ddf2['DR_Avg'] / dcs['ghi']
    idx = t1[t1 > 0.7].index.intersection(t2[t2 > 0.7].index)
    ndf1 = pd.concat([df1[t.strftime('%Y%m%d')] for t in idx])
    ndf2 = pd.concat([df2[t.strftime('%Y%m%d')] for t in idx])
    ncs = pd.concat([cs[t.strftime('%Y%m%d')] for t in idx])
    return ndf1, ndf2, ncs


def select_cloudy(df1, df2, cs):
    ddf1, ddf2, dcs = get_daily(df1, df2, cs)
    t1 = ddf1['Rs_downwell_Avg'] / dcs['ghi']
    t2 = ddf2['DR_Avg'] / dcs['ghi']
    idx = t1[t1 < 0.3].index.intersection(t2[t2 < 0.3].index)
    ndf1 = pd.concat([df1[t.strftime('%Y%m%d')] for t in idx])
    ndf2 = pd.concat([df2[t.strftime('%Y%m%d')] for t in idx])
    ncs = pd.concat([cs[t.strftime('%Y%m%d')] for t in idx])
    return ndf1, ndf2, ncs


def select_day(df1, df2, cs):
    THRESHOLD = 1
    idx = cs.loc[cs['ghi'] > THRESHOLD].index
    idx = idx.intersection(df1.index).intersection(df2.index)
    return df1.loc[idx], df2.loc[idx], cs.loc[idx]


def select_night(df1, df2, cs):
    THRESHOLD = 1
    idx = cs.loc[cs['ghi'] < THRESHOLD].index
    idx = idx.intersection(df1.index).intersection(df2.index)
    return df1.loc[idx], df2.loc[idx], cs.loc[idx]


def show_data(x, *ys, xlabel='', ylabel='', title='', marker=None, grid=False):
    fig, ax = plt.subplots()
    for y in ys:
        ax.plot(x, y, marker=marker)
    ax.legend()
    ax.axhline(color="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    plt.title(title)
    plt.grid(grid)
    plt.show()


def plot_sigma(sigma):
    fig, ax = plt.subplots()
    ax.plot(sigma, marker='o')
    ax.set_xticks(range(len(sigma)))
    ax.annotate('{txt:.2f}'.format(txt=sigma[0]),
                (0.4, sigma[0] - 50))
    for i in range(1, len(sigma)):
        ax.annotate('{txt:.2f}'.format(txt=sigma[i]),
                    (i, sigma[i] + 250), rotation=35)
    plt.show()


def show_compare2(x1, y1, x2, y2, xlabel='', ylabel='', title='', marker=None):
    fig = plt.figure(1)
    ax1 = plt.subplot(211)
    ax1.plot(x1, y1, color="blue", label='USTC', marker=marker)
    ax1.axhline(color="black")
    ax1.legend()
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(x2, y2, color="red", label='HFCAS', marker=marker)
    ax2.axhline(color="black")
    ax2.legend()
    fig.autofmt_xdate()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show


def plot_compare(x1, y1, x2, y2, axhline=True, ylim=None, marker=None,
                 xlabel='', ylabel='', title='', grid=True):
    fig, ax = plt.subplots()
    ax.plot(x2, y2, color='red', marker=marker, label='HFCAS')
    ax.plot(x1, y1, color='blue', marker=marker, label='USTC', alpha=0.75)
    ax.legend()  # (loc='upper right')
    ax.set_ylim(ylim)
    if axhline:
        ax.axhline(color="black")
    fig.autofmt_xdate()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)


def show_compare(x1, y1, x2, y2, ylim=None, marker=None,
                 xlabel='', ylabel='', title='', grid=True):
    plot_compare(x1, y1, x2, y2, ylim=ylim, marker=marker,
                 xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
    plt.show()


def plot_compare3(x1, y1, x2, y2, x3, y3, legend_loc=None, grid=True,
                  xlabel='', ylabel='', title='', marker=None,
                  axhline=True):
    fig, ax = plt.subplots()
    ax.plot(x3, y3, color='green', label='Theory', marker=marker)
    ax.plot(x2, y2, color='red', label='HFCAS', alpha=0.75, marker=marker)
    ax.plot(x1, y1, color='blue', label='USTC', alpha=0.75, marker=marker)
    ax.legend(loc=legend_loc)  # (loc='upper right')
    if axhline:
        ax.axhline(color="black")
    fig.autofmt_xdate()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)


def plot_wind(df, title='Wind'):
    WS_Avg = df['wnd_spd']
    WD_SD = df['wnd_dir_compass']
    TIMESTAMP = df.index
    fig, ax1 = plt.subplots()
    ax1.plot(TIMESTAMP, WS_Avg, 'b-')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Wind speed', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(TIMESTAMP, WD_SD, 'r.')
    ax2.set_ylabel('Wind direction', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(0, 360)
    legend_speed = mpatches.Patch(color='b', label='Wind speed')
    legend_direction = mpatches.Patch(color='r', label='Wind direction')
    plt.legend(handles=[legend_speed, legend_direction],
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.title(title)
    fig.autofmt_xdate()


def plot_Rl_Ta(df1, df2, title=r'$R_l\uparrow$ and Temperature'):
    fig, ax1 = plt.subplots()
    ax1.plot(df1.index, df1['Rl_upwell_Avg'], 'b-')
    ax1.plot(df1.index, df2['ULR_Avg'], 'r-')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Irradiance $W/m^2$')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(df1.index, df1['Ta_2m_Avg'], 'b--')
    ax2.plot(df2.index, df2['Ta_1_Avg'], 'r--')
    ax2.set_ylabel(r'Temperature $(\,^{\circ}C)$')
    ax2.tick_params('y')
    legend_USTC = mpatches.Patch(color='b', label='USTC')
    legend_HFCAS = mpatches.Patch(color='r', label='HFCAS')
    plt.legend(handles=[legend_USTC, legend_HFCAS],
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.title(title)
    fig.autofmt_xdate()
    plt.grid()


def plot_DLR_Ta(df1, df2, title=r'$R_l\downarrow$ and Temperature'):
    fig, ax1 = plt.subplots()
    ax1.plot(df1.index, df1['Rl_downwell_Avg'], 'b-')
    ax1.plot(df1.index, df2['DLR_Avg'], 'r-')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Irradiance $W/m^2$')
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(df1.index, df1['Ta_2m_Avg'], 'b--')
    ax2.plot(df2.index, df2['Ta_1_Avg'], 'r--')
    ax2.set_ylabel(r'Temperature $(\,^{\circ}C)$')
    ax2.tick_params('y')
    legend_USTC = mpatches.Patch(color='b', label='USTC')
    legend_HFCAS = mpatches.Patch(color='r', label='HFCAS')
    plt.legend(handles=[legend_USTC, legend_HFCAS],
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.title(title)
    fig.autofmt_xdate()
    plt.grid()


def show_compare3(x1, y1, x2, y2, x3, y3, legend_loc=None, grid=True,
                  xlabel='', ylabel='', title='', marker=None):
    plot_compare3(x1, y1, x2, y2, x3, y3, legend_loc=legend_loc,
                  xlabel=xlabel, ylabel=ylabel, title=title, marker=marker)
    plt.show()


def save_figs(df1, df2, cs, prefix='', suffix='', marker=None,
              resample=None):
    if resample is not None:
        df1 = df1.resample(resample).mean()
        df2 = df2.resample(resample).mean()
        cs = cs.resample(resample).mean()
    plot_compare3(df1.index, df1['Rs_downwell_Avg'], df2.index, df2['DR_Avg'],
                  cs.index, cs.ghi, marker=marker,
                  xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                  title='Downward shortwave irradiance')
    plt.savefig('{}DSR{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Rs_upwell_Avg'], df2.index, df2['UR_Avg'],
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 marker=marker,
                 title='Upward shortwave irradiance')
    plt.savefig('{}USR{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['albedo'], df2.index, df2['albedo'],
                 marker=marker,
                 xlabel='Time', ylabel='Albedo', title='Albedo')
    plt.savefig('{}albedo{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Rl_downwell_Avg'], df2.index, df2['DLR_Avg'],
                 axhline=False, marker=marker,
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Downward longwave irradiance')
    plt.savefig('{}DLR{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Uz_Avg'], df2.index, df2['Uz_Avg'],
                 marker=marker,
                 xlabel='Time', ylabel='Uz (m/s)', title='Uz')
    plt.savefig('{}Uz{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Rl_upwell_Avg'], df2.index, df2['ULR_Avg'],
                 axhline=False, marker=marker,
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Upward longwave irradiance')
    plt.savefig('{}ULR{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Rn'], df2.index, df2['Rn'], marker=marker,
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Net irradiance')
    plt.savefig('{}Rn{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Hs'], df2.index, df2['Hs'], marker=marker,
                 xlabel='Time', ylabel='Hs $(W/m^2)$',
                 title='Hs')
    plt.savefig('{}Hs{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['LE_irga'], df2.index, df2['LE_irga'],
                 xlabel='Time', ylabel='LE $(W/m^2)$', marker=marker,
                 title='LE')
    plt.savefig('{}LE{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Ta_2m_Avg'], df2.index, df2['Ta_1_Avg'],
                 xlabel='Time', ylabel=r'Temperature $(\,^{\circ}C)$',
                 marker=marker,
                 axhline=False, title='Temperature')
    plt.savefig('{}Ta{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['Ta_10m_Avg'], df2.index, df2['Ta_5_Avg'],
                 xlabel='Time', ylabel=r'Temperature $(\,^{\circ}C)$',
                 marker=marker,
                 axhline=False, title='Temperature')
    plt.savefig('{}Ta_high{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['dS'], df2.index, df2['dS'], marker=marker,
                 xlabel='Time', ylabel="$\Delta S'$ $(W/m^2)$",
                 title="$\Delta S'$")
    plt.savefig('{}dS{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['RH_2m_Avg'], df2.index, df2['RH_1_Avg'],
                 xlabel='Time', ylabel='RH (%)', marker=marker,
                 title='RH')
    plt.savefig('{}RH{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['RH_10m_Avg'], df2.index, df2['RH_5_Avg'],
                 xlabel='Time', ylabel='RH (%)', marker=marker,
                 title='RH')
    plt.savefig('{}RH_high{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_wind(df1, title='USTC Wind')
    plt.savefig('{}wnd_USTC{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_wind(df2, title='HFCAS Wind')
    plt.savefig('{}wnd_HFCAS{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_compare(df1.index, df1['wnd_spd'], df2.index, df2['wnd_spd'],
                 marker=marker,
                 xlabel='Time', ylabel='Speed $(m/s)$', title='Wind speed')
    plt.savefig('{}wnd_spd{}.pdf'.format(prefix, suffix))
    plt.close()
    plot_Rl_Ta(df1, df2)
    plt.savefig('{}Rl_Ta{}.pdf'.format(prefix, suffix))
    plt.close()


def save_semi_month_figs(df1, df2, cs, path):
    df1, df2, cs = select_date(df1, df2, cs, '20151101', '20160531')
    sdf1, sdf2, scs = select_sunny(df1, df2, cs)
    cdf1, cdf2, ccs = select_cloudy(df1, df2, cs)
    ddf1, ddf2, dcs = select_day(df1, df2, cs)
    ndf1, ndf2, ncs = select_night(df1, df2, cs)
    dsdf1, dsdf2, dscs = select_day(sdf1, sdf2, scs)
    dcdf1, dcdf2, dccs = select_day(cdf1, cdf2, ccs)
    nsdf1, nsdf2, nscs = select_night(sdf1, sdf2, scs)
    ncdf1, ncdf2, nccs = select_night(cdf1, cdf2, ccs)
    save_figs(df1, df2, cs, prefix=path+'sms_', resample='SMS', marker='o')
    save_figs(sdf1, sdf2, scs, prefix=path+'sms_s_',
              resample='SMS', marker='o')
    save_figs(cdf1, cdf2, ccs, prefix=path+'sms_c_',
              resample='SMS', marker='o')
    save_figs(ddf1, ddf2, dcs, prefix=path+'sms_d_',
              resample='SMS', marker='o')
    save_figs(ndf1, ndf2, ncs, prefix=path+'sms_n_',
              resample='SMS', marker='o')
    save_figs(dsdf1, dsdf2, dscs, prefix=path+'sms_ds_',
              resample='SMS', marker='o')
    save_figs(dcdf1, dcdf2, dccs, prefix=path+'sms_dc_',
              resample='SMS', marker='o')
    save_figs(nsdf1, nsdf2, nscs, prefix=path+'sms_ns_',
              resample='SMS', marker='o')
    save_figs(ncdf1, ncdf2, nccs, prefix=path+'sms_nc_',
              resample='SMS', marker='o')


def plot_compare_sms(df1, df2, df1_label, df2_label,
                     xlabel='', ylabel='', title=''):
    df1, df2 = df1['20151101':'20160531'], df2['20151101':'20160531']
    fig, ax = plt.subplots()
    ax.errorbar(df1.resample('SMS').mean().index,
                df1.resample('SMS').mean()[df1_label],
                df1.resample('SMS').std()[df1_label],
                fmt='bo-', alpha=0.75, label='USTC')
    ax.errorbar(df2.resample('SMS').mean().index,
                df2.resample('SMS').mean()[df2_label],
                df2.resample('SMS').std()[df2_label],
                fmt='ro-', alpha=0.75, label='HFCAS')
    ax.legend()
    fig.autofmt_xdate()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)


def show_compare_sms(df1, df2, df1_label, df2_label,
                     xlabel='', ylabel='', title=''):
    plot_compare_sms(df1, df2, df1_label, df2_label,
                     xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()


def print_org_mode(df1, df2, cs, path, start_date, end_date):
    ndf1, ndf2, ncs = select_date(df1, df2, cs, start_date, end_date)
    date_string = '{}_{}_'.format(start_date, end_date)
    save_figs(ndf1, ndf2, ncs, prefix=path+date_string)
    org_mode = '''
** 向下短波辐射

   #+attr_latex: :float t
   [[file:resources/{0}DSR.pdf]]

** 向上短波辐射

   #+attr_latex: :float t
   [[file:resources/{0}USR.pdf]]

** 反照率

   #+attr_latex: :float t
   [[file:resources/{0}albedo.pdf]]

** 向下长波辐射

   #+attr_latex: :float t
   [[file:resources/{0}DLR.pdf]]

** 垂直风速

   #+attr_latex: :float t
   [[file:resources/{0}Uz.pdf]]

** CO_2

   #+attr_latex: :float t
   [[file:resources/{0}CO2.pdf]]

** H_{{2}}O

   #+attr_latex: :float t
   [[file:resources/{0}H2O.pdf]]

** 向上长波辐射

   #+attr_latex: :float t
   [[file:resources/{0}ULR.pdf]]

** 温度

   #+attr_latex: :float t
   [[file:resources/{0}Ta.pdf]]

** 向上长波辐射与温度

   #+attr_latex: :float t
   [[file:resources/{0}Rl_Ta.pdf]]

** 净辐射

   #+attr_latex: :float t
   [[file:resources/{0}Rn.pdf]]

** 感热

   #+attr_latex: :float t
   [[file:resources/{0}Hs.pdf]]

** 潜热

   #+attr_latex: :float t
   [[file:resources/{0}LE.pdf]]

** \Delta S

   #+attr_latex: :float t
   [[file:resources/{0}dS.pdf]]

** RH

   #+attr_latex: :float t
   [[file:resources/{0}RH.pdf]]

** 风(USTC)

   #+attr_latex: :float t
   [[file:resources/{0}wnd_USTC.pdf]]

** 风(HFCAS)

   #+attr_latex: :float t
   [[file:resources/{0}wnd_HFCAS.pdf]]

** 风速比较

   #+attr_latex: :float t
   [[file:resources/{0}wnd_spd.pdf]]

'''.format(date_string)
    return org_mode


def get_summary(df1, df2, label1, label2):
    d = df2[label2] - df1[label1]
    cmidx = df1[label1].dropna().index.intersection(df2[label2].dropna().index)
    r, p = stats.pearsonr(df1.loc[cmidx, label1], df2.loc[cmidx, label2])
    return (df1[label1].mean(), df1[label1].std(),
            df2[label2].mean(), df2[label2].std(),
            r, p,
            d.mean(), d.std())


def org_summary(df1, df2, cs, path):
    save_figs(df1, df2, cs, prefix=path)
    org_mode = '''
** 向下短波辐射通量密度

   #+attr_latex: :float t
   [[file:resources/DSR.pdf]]

** 向上短波辐射通量密度

   #+attr_latex: :float t
   [[file:resources/USR.pdf]]

** 反照率

   #+attr_latex: :float t
   [[file:resources/albedo.pdf]]

** 向下长波辐射通量密度

   #+attr_latex: :float t
   [[file:resources/DLR.pdf]]

** 向上长波辐射通量密度

   #+attr_latex: :float t
   [[file:resources/ULR.pdf]]

** 净辐射通量密度

   #+attr_latex: :float t
   [[file:resources/Rn.pdf]]

** 感热通量密度

   #+attr_latex: :float t
   [[file:resources/Hs.pdf]]

** 潜热通量密度

   #+attr_latex: :float t
   [[file:resources/LE.pdf]]

** 垂直风速

   #+attr_latex: :float t
   [[file:resources/Uz.pdf]]

** 相对湿度

   #+attr_latex: :float t
   [[file:resources/RH.pdf]]

** 温度

   #+attr_latex: :float t
   [[file:resources/Ta.pdf]]

** 风速

   #+attr_latex: :float t
   [[file:resources/wnd_spd.pdf]]

** CO_2

   #+attr_latex: :float t
   [[file:resources/CO2.pdf]]

** H_{2}O

   #+attr_latex: :float t
   [[file:resources/H2O.pdf]]

    '''
    return org_mode


def table_summary(df1, df2):
    table_header = '''
   |---+------+-----+-------+-----+------+---------+------+-----|
   |   | USTC |     | HFCAS |     |      |         | d    |     |
   |---+------+-----+-------+-----+------+---------+------+-----|
   |   | mean | std | mean  | std | corr | p-value | mean | std |
   |---+------+-----+-------+-----+------+---------+------+-----|
   | / | <    |     | <     |     | <    |         | <    |     |
'''
    table_foot = '   |---+------+-----+-------+-----+' +\
                 '------+---------+------------+-----|\n'
    fmt = '   | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n'
    table1_rows = [fmt % (('DSR',) +
                          get_summary(df1, df2, 'Rs_downwell_Avg', 'DR_Avg')),
                   fmt % (('USR',) +
                          get_summary(df1, df2, 'Rs_upwell_Avg', 'UR_Avg')),
                   fmt % (('Albedo',) +
                          get_summary(df1, df2, 'albedo', 'albedo')),
                   fmt % (('DLR',) +
                          get_summary(df1, df2, 'Rl_downwell_Avg', 'DLR_Avg')),
                   fmt % (('ULR',) +
                          get_summary(df1, df2, 'Rl_upwell_Avg', 'ULR_Avg')),
                   fmt % (('Rn',) +
                          get_summary(df1, df2, 'Rn', 'Rn')),
                   fmt % (('Hs',) +
                          get_summary(df1, df2, 'Hs', 'Hs'))]
    table2_rows = [fmt % (('LE',) +
                          get_summary(df1, df2, 'LE_irga', 'LE_irga')),
                   fmt % (('Uz',) +
                          get_summary(df1, df2, 'Uz_Avg', 'Uz_Avg')),
                   fmt % (('RH',) +
                          get_summary(df1, df2, 'RH_10m_Avg', 'RH_5_Avg')),
                   fmt % (('Ta',) +
                          get_summary(df1, df2, 'Ta_2m_Avg', 'Ta_1_Avg')),
                   fmt % (('=wnd_spd=',) +
                          get_summary(df1, df2, 'wnd_spd', 'wnd_spd')),
                   fmt % (('CO_2',) +
                          get_summary(df1, df2, 'CO2_mean', 'co2_Avg')),
                   fmt % (('H_{2}O',) +
                          get_summary(df1, df2, 'H2O_mean', 'h2o_Avg'))]
    table1 = table_header + ''.join(table1_rows) + table_foot
    table2 = table_header + ''.join(table2_rows) + table_foot
    return table1, table2


def save_20160525_DSR(rdf1, rdf2, df1, df2, cs, path):
    rndf1, rndf2, ncs = select_date(rdf1, rdf2, cs, '20160525', '20160526')
    ndf1, ndf2, ncs = select_date(df1, df2, cs, '20160525', '20160526')
    plot_compare3(rndf1.index, rndf1['Rs_downwell_Avg'],
                  rndf2.index, rndf2['DR_Avg'],
                  ncs.index, ncs['ghi'],
                  xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                  title='Downward shortwave irradiance')
    plt.savefig(path + '20160525_DSR_raw.pdf')
    plt.close()
    plot_compare3(ndf1.index, ndf1['Rs_downwell_Avg'],
                  ndf2.index, ndf2['DR_Avg'],
                  ncs.index, ncs['ghi'],
                  xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                  title='Downward shortwave irradiance')
    plt.savefig(path + '20160525_DSR.pdf')
    plt.close()


def save_DSR_Hs_LE(rdf1, rdf2, df1, df2, path):
    plot_compare(rdf1.index, rdf1['Rs_downwell_Avg'],
                 rdf2.index, rdf2['DR_Avg'],
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Downward shortwave irradiance')
    plt.savefig(path + 'raw_DSR.pdf')
    plt.close()
    plot_compare(df1.index, df1['Rs_downwell_Avg'], df2.index, df2['DR_Avg'],
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Downward shortwave irradiance')
    plt.savefig(path + 'DSR.pdf')
    plt.close()
    plot_compare(df1.index, df1['Hs'], df2.index, df2['Hs'],
                 xlabel='Time', ylabel='Hs $(W/m^2)$',
                 title='Hs')
    plt.savefig(path + 'Hs.pdf')
    plt.close()
    plot_compare(rdf1.index, rdf1['LE_irga'], rdf2.index, rdf2['LE_irga'],
                 xlabel='Time', ylabel='LE $(W/m^2)$', title='LE',
                 ylim=(-210, 510))
    plt.savefig(path + 'raw_LE.pdf')
    plt.close()
    plot_compare(df1.index, df1['LE_irga'], df2.index, df2['LE_irga'],
                 xlabel='Time', ylabel='LE $(W/m^2)$',
                 title='LE')
    plt.savefig(path + 'LE.pdf')
    plt.close()


def save_USR(df1, df2, cs, path):
    plot_compare(df1.index, df1['Rs_upwell_Avg'],
                 df2.index, df2['UR_Avg'],
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Upward shortwave irradiance')
    plt.savefig(path + 'USR.pdf')
    plt.close()


def save_20160201(df1, df2, cs, path):
    ndf1, ndf2, ncs = select_date(df1, df2, cs, '20160128', '20160205')
    plot_compare3(ndf1.index, ndf1['Rs_downwell_Avg'],
                  ndf2.index, ndf2['DR_Avg'],
                  ncs.index, ncs['ghi'],
                  xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                  title='Downward shortwave irradiance')
    plt.savefig(path + '20160201_DSR.pdf')
    plt.close()
    plot_compare(ndf1.index, ndf1['Rs_upwell_Avg'],
                 ndf2.index, ndf2['UR_Avg'],
                 xlabel='Time', ylabel='Irradiance $(W/m^2)$',
                 title='Upward shortwave irradiance')
    plt.savefig(path + '20160201_USR.pdf')
    plt.close()
    plot_compare(ndf1.index, ndf1['albedo'],
                 ndf2.index, ndf2['albedo'],
                 xlabel='Time', ylabel='Albedo',
                 title='Albedo')
    plt.savefig(path + '20160201_albedo.pdf')
    plt.close()


def save_figs_date(df1, df2, cs, path, start_date, end_date):
    ndf1, ndf2, ncs = select_date(df1, df2, cs, start_date, end_date)
    date_string = '{}_{}_'.format(start_date, end_date)
    save_figs(ndf1, ndf2, ncs, prefix=path+date_string)


def save_ULR_Theory(df1, df2, cs, path, start_date, end_date):
    df1['ULR_Theory'] = (5.67*10**(-8))*(df1['Ta_2m_Avg']+273.15)**4
    df2['ULR_Theory'] = (5.67*10**(-8))*(df2['Ta_1_Avg']+273.15)**4
    ndf1, ndf2, ncs = select_date(df1, df2, cs, start_date, end_date)
    fig, ax = plt.subplots()
    ax.plot(ndf1.index, ndf1['Rl_upwell_Avg'], color='b', label='Observed')
    ax.plot(ndf1.index, ndf1['ULR_Theory'], color='r', label='Theory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance $(W/m^2)$')
    ax.legend()
    fig.autofmt_xdate()
    plt.title('Upward longwave irradiance')
    plt.grid()
    plt.savefig(path +
                '{}_{}_ULR_Theory_USTC.pdf'.format(start_date, end_date))
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(ndf2.index, ndf2['ULR_Avg'], color='b', label='Observed')
    ax.plot(ndf2.index, ndf2['ULR_Theory'], color='r', label='Theory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance $(W/m^2)$')
    ax.legend()
    fig.autofmt_xdate()
    plt.title('Upward longwave irradiance')
    plt.grid()
    plt.savefig(path +
                '{}_{}_ULR_Theory_HFCAS.pdf'.format(start_date, end_date))
    plt.close()


def save_sms_R(df1, df2, cs, path):
    df1, df2, cs = select_date(df1, df2, cs, '20151101', '20160531')
    df1 = df1.resample('SMS').mean()
    df2 = df2.resample('SMS').mean()
    cs = cs.resample('SMS').mean()
    fig, ax = plt.subplots()
    ax.plot(cs.index, cs['ghi'], 'g--v',
            label=r'$R_{s}\downarrow$ in Theory')
    ax.plot(df1.index, df1['Rs_downwell_Avg'], 'b--v',
            label=r'USTC $R_{s}\downarrow$')
    ax.plot(df2.index, df2['DR_Avg'], 'r--v',
            label=r'HFCAS $R_{s}\downarrow$')
    ax.plot(df1.index, df1['Rs_upwell_Avg'], 'b--^',
            label=r'USTC $R_{s}\uparrow$')
    ax.plot(df2.index, df2['UR_Avg'], 'r--^',
            label=r'HFCAS $R_{s}\uparrow$')
    ax.plot(df1.index, df1['Rl_downwell_Avg'], 'b-v',
            label=r'USTC $R_{l}\downarrow$')
    ax.plot(df2.index, df2['DLR_Avg'], 'r-v',
            label=r'HFCAS $R_{l}\downarrow$')
    ax.plot(df1.index, df1['Rl_upwell_Avg'], 'b-^',
            label=r'USTC $R_{l}\uparrow$')
    ax.plot(df2.index, df2['ULR_Avg'], 'r-^',
            label=r'HFCAS $R_{l}\uparrow$')
    ax.plot(df1.index, df1['Rn'], 'b:*',
            label='USTC $R_n$')
    ax.plot(df2.index, df2['Rn'], 'r:*',
            label='HFCAS $R_n$')
    legend = ax.legend(bbox_to_anchor=(1, 1))
    ax.axhline(color="black")
    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance $(W/m^2)$')
    fig.autofmt_xdate()
    plt.title('Semimonthly average Irradiance')
    plt.grid()
    plt.savefig(path + 'sms_R.pdf',
                bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.close()


def save_Gs(df2, path):
    fig, ax = plt.subplots()
    ax.plot(df2.index, df2['Rn'], 'r', label='$R_n$')
    ax.plot(df2.index, df2['G_Avg'], 'g', label='$G_s$')
    ax.axhline(color="black")
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux density $(W/m^2)$')
    fig.autofmt_xdate()
    plt.title('Net irradiance $R_n$ and $G_s$ in HFCAS')
    plt.grid()
    plt.savefig(path + 'Rn_Gs.pdf')
    plt.close()


def save_dS(df1, df2, cs, path):
    plot_compare(df1.index, df1['dS'], df2.index, df2['dS'],
                 xlabel='Time', ylabel="$\Delta S'$", title="$\Delta S'$")
    plt.savefig(path + 'dS.pdf')
    plt.close()
    df1, df2, cs = select_date(df1, df2, cs, '20151101', '20160531')
    df1 = df1.resample('SMS').mean()
    df2 = df2.resample('SMS').mean()
    plot_compare(df1.index, df1['dS'], df2.index, df2['dS'],
                 xlabel='Time', ylabel="$\Delta S'$", marker='o',
                 title="Semimonthly average $\Delta S'$")
    plt.savefig(path + 'dS_sms.pdf')
    plt.close()
    # fig, ax = plt.subplots()
    # ax.plot(df1.index, df1['Rn'], 'bv-', label='USTC $R_n$')
    # ax.plot(df2.index, df2['Rn'], 'rv-', label='HFCAS $R_n$')
    # ax.plot(df1.index, df1['Hs'], 'b^-', label='USTC $H_s$')
    # ax.plot(df2.index, df2['Hs'], 'r^-', label='HFCAS $H_s$')
    # ax.plot(df1.index, df1['LE_irga'], 'bo-', label='USTC LE')
    # ax.plot(df2.index, df2['LE_irga'], 'ro-', label='HFCAS LE')
    # ax.plot(df1.index, df1['dS'], 'b*-', label=r'USTC $\Delta S$')
    # ax.plot(df2.index, df2['dS'], 'r*-', label=r'HFCAS $\Delta S$')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Radiation flux density $(W/m^2)$')
    # fig.autofmt_xdate()
    # ax.legend()
    # plt.title('Radiation flux density')
    # plt.savefig(path + 'dS_sms_all.pdf')
    # plt.close()


def save_dS_Q(df1, df2, cs, path):
    df1, df2, cs = select_date(df1, df2, cs, '20151101', '20160531')
    df1.loc['20151115':'20160315',
            'dS'] = df1.loc['20151115':'20160315', 'dS'] + 30
    df1, df2 = df1.resample('SMS').mean(), df2.resample('SMS').mean()
    plot_compare(df1.index, df1['dS'], df2.index, df2['dS'],
                 xlabel='Time', ylabel="$\Delta S''$", marker='o',
                 title="Semimonthly average $\Delta S''$")
    plt.savefig(path + 'dS_Q.pdf')
    plt.close()


def save_DLR(df1, df2, cs, path):
    df1, df2, cs = select_date(df1, df2, cs, '20151101', '20160531')
    fig, ax = plt.subplots()
    ax.errorbar(df1.resample('SMS').mean().index,
                df1.resample('SMS').mean()['Rl_downwell_Avg'],
                df1.resample('SMS').std()['Rl_downwell_Avg'],
                fmt='bo-', alpha=0.75, label='USTC')
    ax.errorbar(df2.resample('SMS').mean().index,
                df2.resample('SMS').mean()['DLR_Avg'],
                df2.resample('SMS').std()['DLR_Avg'],
                fmt='ro-', alpha=0.75, label='HFCAS')
    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance $(W/m^2)$')
    fig.autofmt_xdate()
    plt.title('Semimonthly average downward longwave irradiance')
    plt.savefig(path + 'DLR_sms_std.pdf')
    plt.close()


def save_co2(rdf2, df2, path):
    fig, ax = plt.subplots()
    ax.plot(rdf2.index, rdf2['co2_Avg'])
    ax.set_xlabel('Time')
    ax.set_ylabel('CO_2 $(mg/m^3)$')
    fig.autofmt_xdate()
    plt.title('$CO_2$')
    plt.grid()
    plt.savefig(path + 'raw_CO2.pdf')
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(df2.index, df2['co2_Avg'])
    ax.set_xlabel('Time')
    ax.set_ylabel('CO_2 $(mg/m^3)$')
    ax.set_ylim(690, 910)
    fig.autofmt_xdate()
    plt.title('$CO_2$')
    plt.grid()
    plt.savefig(path + 'CO2.pdf')
    plt.close()


def save_all(rdf1, rdf2, df1, df2, cs, path):
    save_20160525_DSR(rdf1, rdf2, df1, df2, cs, path)
    save_DSR_Hs_LE(rdf1, rdf2, df1, df2, path)
    save_USR(df1, df2, cs, path)
    save_20160201(df1, df2, cs, path)
    save_figs_date(df1, df2, cs, path, '20160121', '20160126')
    save_figs_date(df1, df2, cs, path, '20160215', '20160219')
    save_figs_date(df1, df2, cs, path, '20160426', '20160430')
    save_figs_date(df1, df2, cs, path, '20160429', '20160505')
    save_ULR_Theory(df1, df2, cs, path, '20151201', '20151231')
    # save_semi_month_figs(df1, df2, cs, path)
    save_sms_R(df1, df2, cs, path)
    # save_Gs(df2, path)
    save_dS(df1, df2, cs, path)
    save_co2(rdf2, df2, path)


def latex_table(df1, df2, cs, label1, label2):
    def flatten(l):
        return tuple([item for sublist in l for item in sublist])
    sdf1, sdf2, scs = select_sunny(df1, df2, cs)
    cdf1, cdf2, ccs = select_cloudy(df1, df2, cs)
    ddf1, ddf2, dcs = select_day(df1, df2, cs)
    ndf1, ndf2, ncs = select_night(df1, df2, cs)
    dsdf1, dsdf2, dscs = select_day(sdf1, sdf2, scs)
    dcdf1, dcdf2, dccs = select_day(cdf1, cdf2, ccs)
    nsdf1, nsdf2, nscs = select_night(sdf1, sdf2, scs)
    ncdf1, ncdf2, nccs = select_night(cdf1, cdf2, ccs)
    data = (df1, df2, sdf1, sdf2, cdf1, cdf2,
            ddf1, ddf2, ndf1, ndf2,
            dsdf1, dsdf2, dcdf1, dcdf2,
            nsdf1, nsdf2, ncdf1, ncdf2)
    latex = '''
  全部 & %.3f & %.3f & %.3f & %.3f\\\\
  晴天 & %.3f & %.3f & %.3f & %.3f\\\\
  阴天 & %.3f & %.3f & %.3f & %.3f\\\\
  白天 & %.3f & %.3f & %.3f & %.3f\\\\
  夜晚 & %.3f & %.3f & %.3f & %.3f\\\\
  晴天白天 & %.3f & %.3f & %.3f & %.3f\\\\
  阴天白天 & %.3f & %.3f & %.3f & %.3f\\\\
  晴天夜晚 & %.3f & %.3f & %.3f & %.3f\\\\
  阴天夜晚 & %.3f & %.3f & %.3f & %.3f\\\\
''' % flatten([(t[label1 if 'Rs_downwell_Avg' in t else label2].mean(),
                t[label1 if 'Rs_downwell_Avg' in t else label2].std())
               for t in data])
    return latex


def org_table(df1, df2, cs, label1, label2):
    def flatten(l):
        return tuple([item for sublist in l for item in sublist])
    sdf1, sdf2, scs = select_sunny(df1, df2, cs)
    cdf1, cdf2, ccs = select_cloudy(df1, df2, cs)
    ddf1, ddf2, dcs = select_day(df1, df2, cs)
    ndf1, ndf2, ncs = select_night(df1, df2, cs)
    dsdf1, dsdf2, dscs = select_day(sdf1, sdf2, scs)
    dcdf1, dcdf2, dccs = select_day(cdf1, cdf2, ccs)
    nsdf1, nsdf2, nscs = select_night(sdf1, sdf2, scs)
    ncdf1, ncdf2, nccs = select_night(cdf1, cdf2, ccs)
    data = (df1, df2, sdf1, sdf2, cdf1, cdf2,
            ddf1, ddf2, ndf1, ndf2,
            dsdf1, dsdf2, dcdf1, dcdf2,
            nsdf1, nsdf2, ncdf1, ncdf2)
    org = '''
   | 全部 | %.1f | %.1f | %.1f | %.1f |
   | 晴天 | %.1f | %.1f | %.1f | %.1f |
   | 阴天 | %.1f | %.1f | %.1f | %.1f |
   | 白天 | %.1f | %.1f | %.1f | %.1f |
   | 夜晚 | %.1f | %.1f | %.1f | %.1f |
   | 晴天白天 | %.1f | %.1f | %.1f | %.1f |
   | 阴天白天 | %.1f | %.1f | %.1f | %.1f |
   | 晴天夜晚 | %.1f | %.1f | %.1f | %.1f |
   | 阴天夜晚 | %.1f | %.1f | %.1f | %.1f |
''' % flatten([(t[label1 if 'Rs_downwell_Avg' in t else label2].mean(),
                t[label1 if 'Rs_downwell_Avg' in t else label2].std())
               for t in data])
    return org
