import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

# 设置 Times New Roman 字体和处理负号显示问题
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 常数定义
sigma = 5.670374419e-8  # Stefan-Boltzmann 常数, W·m⁻²·K⁻⁴
G_sc = 1353  # 太阳常数, W/m²
H = 8500  # scale height, 单位：m

# 站点海拔数据（单位：m）
station_altitudes = {
    'BON': 213,
    'DRA': 1007,
    'FPK': 634,  # FPK 对应论文中的 FRK
    'GWN': 98,
    'PSU': 376,
    'SXF': 473,
    'TBL': 1689
}

# 各站点的拟合参数，用于函数 CF = c1 * k_d^(c2)
station_params = {
    'BON': (0.625, 1.913),
    'DRA': (0.348, 1.230),
    'FPK': (0.593, 1.531),
    'GWN': (0.610, 1.622),
    'PSU': (0.656, 2.054),
    'SXF': (0.661, 1.893),
    'TBL': (0.531, 1.819)
}

# 总体拟合参数
overall_c1 = 0.597
overall_c2 = 1.776

# 数据文件路径
precipitation_h5_path = r'C:\Users\jiw181\PycharmProjects\pythonProject1\testing_focus_on_altitude_correction\precipitation_data\all_precipitation_data.h5'
file_path = r'C:\Users\jiw181\PycharmProjects\pythonProject1\output.h5'
stations = ['BON', 'DRA', 'FPK', 'GWN', 'PSU', 'SXF', 'TBL']

# 用于存储各站点处理后的 DataFrame，后面用于绘制全站散点图
data_list = []

# 读取降雨数据
with pd.HDFStore(precipitation_h5_path, 'r') as precip_store:
    precipitation_data = {
        station: precip_store[station].set_index('datetime')['PRECIPITATIONCAL']
        for station in stations
    }

# 定义输出图像保存目录
output_dir = r'C:\Users\jiw181\PycharmProjects\pythonProject1\2009-2023data_process\scatter_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 定义各站点数据处理函数
def process_station_data(station):
    with pd.HDFStore(file_path, 'r') as h5_file:
        dataset = h5_file[station].copy()
    dataset.index = pd.to_datetime(dataset.index)

    # 对 DRA 站点，删除 2017～2023 年的数据
    if station == 'DRA':
        dataset = dataset[~((dataset.index.year >= 2017) & (dataset.index.year <= 2023))]
    dataset = dataset[~(dataset.index.year == 2023)]

    # 对齐降雨数据：只保留日均降雨量为 0 的日期数据
    precip_data = precipitation_data[station]
    daily_precip = precip_data.groupby(precip_data.index.date).mean()
    valid_dates = daily_precip[daily_precip == 0].index
    valid_dates_index = pd.Index(valid_dates)
    dataset = dataset[dataset.index.normalize().isin(valid_dates_index)]

    # 计算 G_on（日照常数）
    day_of_year = dataset.index.dayofyear
    B = (day_of_year - 1) * 360 / 365
    G_on = G_sc * (1.000110 + 0.034221 * np.cos(np.radians(B)) +
                   0.001280 * np.sin(np.radians(B)) +
                   0.000719 * np.cos(2 * np.radians(B)) +
                   0.000077 * np.sin(2 * np.radians(B)))

    # 筛选条件
    condition_1 = (dataset['sza'] < 72.5) & (dataset['ghi_m'] > 0) & (dataset['dhi_m'] > 0) & (
                dataset['ghi_m'] / dataset['ghi_c'] < 1.5)
    condition_2 = dataset.index.time >= pd.to_datetime("08:00").time()
    condition_4 = dataset['ghi_m'] < (1.2 * G_on * np.cos(np.radians(dataset['sza'])) ** 1.2 + 50)
    condition_6 = dataset['dni_m'] < (0.95 * G_on * np.cos(np.radians(dataset['sza'])) ** 0.2 + 10)
    condition_8 = ((dataset['dhi_m'] / dataset['ghi_m']) <= 1)
    condition_9 = ((dataset['ghi_m'] / dataset['ghi_c']) > 0.1)
    dataset = dataset[condition_1 & condition_2 & condition_4 & condition_6 & condition_8 & condition_9]

    # 计算其它参数
    dataset['k_d'] = dataset['dhi_m'] / dataset['ghi_m']
    dataset['cos_theta_z'] = np.cos(np.radians(dataset['sza']))
    dataset['e_s'] = 6.112 * np.exp(17.625 * dataset['temp'] / (dataset['temp'] - 30.11 + 273.15))
    dataset['pw_hpa'] = dataset['e_s'] * dataset['rh'] / 100
    condition_3 = (dataset['pw_hpa'] >= 0) & (dataset['e_s'] >= 0) & (dataset['dlw'] > 0) & (dataset['temp'] <= 90) & (
                dataset['temp'] >= -80)
    condition_7 = (dataset['qc_direct_n'] == 0) & (dataset['qc_dwsolar'] == 0) & (dataset['qc_diffuse'] == 0) & (
                dataset['qc_dwir'] == 0) & (dataset['qc_temp'] == 0) & (dataset['qc_rh'] == 0) & (
                              dataset['qc_pressure'] == 0)
    dataset = dataset[condition_3 & condition_7]

    dataset['sqrt_pw'] = np.sqrt(dataset['pw_hpa'] / 1013.25)
    dataset['altitude'] = station_altitudes[station]

    # 计算 e_sky 与 e_clear_sky
    dataset['e_sky'] = dataset['dlw'] / (sigma * ((dataset['temp'] + 273.15) ** 4))
    dataset['e_clear_sky'] = 0.6 + 1.652 * dataset['sqrt_pw'] + 0.15 * (np.exp(-dataset['altitude'] / H) - 1)

    # 计算实际 CF 值： (e_sky - e_clear_sky) / (1 - e_clear_sky)
    dataset['CF_actual'] = (dataset['e_sky'] - dataset['e_clear_sky']) / (1 - dataset['e_clear_sky'])

    # 筛选 k_d < 2 的数据点
    dataset = dataset[dataset['k_d'] <= 1]

    return dataset

# 分站点处理并绘图（每个站点绘制一张密度图，叠加两条函数曲线）
for station in stations:
    df_station = process_station_data(station)
    data_list.append(df_station)  # 保存用于后续全站图

    # 横轴为 k_d，纵轴为实际 CF 值
    x_vals = df_station['k_d'].values
    y_vals = df_station['CF_actual'].values

    # 使用 hexbin 绘制密度图，使用对数颜色映射
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x_vals, y_vals, gridsize=50, cmap='YlGnBu', mincnt=1, norm=mcolors.LogNorm())

    # 添加颜色条
    plt.colorbar(hb, label='Density (log scale)')

    # 横轴范围（从 0 到当前数据中 k_d 的最大值）
    x_range = np.linspace(0, 1, 2000)

    # 1. 绘制站点拟合曲线：CF = c1 * k_d^(c2)
    c1, c2 = station_params[station]
    y_station_curve = c1 * (x_range ** c2)
    plt.plot(x_range, y_station_curve, color='magenta', lw=2,
             label=f'{station} fit: {c1:.3f} * $k_d$^{c2:.3f}')

    # 2. 绘制总体拟合曲线：CF = 0.638 * k_d^(1.871)
    y_overall_curve = overall_c1 * (x_range ** overall_c2)
    plt.plot(x_range, y_overall_curve, color='cyan', lw=2,
             label=f'Overall fit: {overall_c1:.3f} * $k_d$^{overall_c2:.3f}')

    # 修改坐标标签，其中 y 轴使用 LaTeX 格式显示公式，且 k_d 显示为 k_d
    plt.xlabel(r'$k_d$', fontsize=12)
    plt.ylabel(r'$(\epsilon_{sky} - \epsilon_{\rm clear\ sky})/(1-\epsilon_{\rm clear\ sky})$', fontsize=12)
    plt.title(f'{station} - Density Plot of r vs $k_d$', fontsize=14)
    plt.legend(frameon=False, fontsize=10)

    # 保存当前站点图像
    output_file = os.path.join(output_dir, f'{station}_density_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{station} density plot saved to {output_file}")

# 绘制全站（所有数据）密度图
df_all = pd.concat(data_list)
# 统一筛选 k_d < 2（各站点已筛选，但这里作统一处理）
df_all = df_all[df_all['k_d'] <= 1]
x_all = df_all['k_d'].values
y_all = df_all['CF_actual'].values

plt.figure(figsize=(8, 6))
# 使用 hexbin 绘制全站的密度图
plt.hexbin(x_all, y_all, gridsize=50, cmap='YlGnBu', mincnt=1, norm=mcolors.LogNorm())
plt.colorbar(label='Density (log scale)')

# 横轴范围
x_range_all = np.linspace(0, 1, 2000)
y_overall_curve_all = overall_c1 * (x_range_all ** overall_c2)
plt.plot(x_range_all, y_overall_curve_all, color='cyan', lw=2,
         label=f'Overall fit: {overall_c1:.3f} * $k_d$^{overall_c2:.3f}')

plt.xlabel(r'$k_d$', fontsize=12)
plt.ylabel(r'$(\epsilon_{sky} - \epsilon_{\rm clear\ sky})/(1-\epsilon_{\rm clear\ sky})$', fontsize=12)
plt.title('All Stations - Density Plot of r vs $k_d$', fontsize=14)
plt.legend(frameon=False, fontsize=10)

output_file_all = os.path.join(output_dir, 'AllStations_density_plot.png')
plt.savefig(output_file_all, dpi=300, bbox_inches='tight')
plt.close()
print(f"All stations density plot saved to {output_file_all}")
