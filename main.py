# Name: Kathleen Nguyen, Robert Bao
# Email ID: kn7wz, cb5th
# Date: 2021-2-23
# File: main.py
# The signal analysis code for pedometer analysis

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np

# OH Notes:
# The best way to get data from the phone sensor: put in the pants
# Hold the phone in hands is BAD
# - requires move hand FORWARD and BACKWARD
# - could be difficult
# Find local maxima is okay -- could have errors though


# get the rolling average of a pandas' series
# 滑动均值滤波
def get_rolling_avg(input_series, window_size):
    windows = input_series.rolling(window_size)

    # Create a list of moving averages
    moving_averages = windows.mean().tolist() # tolist()数组转换成列表
    return moving_averages[window_size - 1:] # 返回剩余数据，前window_size个值为nan

'''
thresholding function
x: 一段长度的avg_acc
threshold：用来筛选
stepH：用来表示一步的高度
res: 长度和x相同的0向量，在满足阈值的帧的位置上置为step
'''
# threshold=0.0035 除1000的时候

def threshold_fn(x, threshold=7.8, stepH=0.01):
    # 生成了一个长度为len(x)的0向量
    res = pd.Series(0*len(x), index=x.index)

    t = x.min() + threshold # x.min()就是局部最小值
    bool_meet_thr = x.gt(t) # 用bool向量记录x序列中满足大于t的位置
    res.loc[bool_meet_thr] = stepH # 行用loc操作，每个数据对应的变成了列标签
    return res

# input file: imported phyphox accelerometer -- without g
# also try: walk-10-step-2022-2-24-v2.csv !
# Note: This will not work with data with g. (The value range is different)

window_size = 100 # 窗就是帧的个数，根据步伐的频率决定

dataType = 2

if dataType == 1: # 自带数据
    filename = "data/walk-10-step-2022-3-2-v1.csv"
    df = pd.read_csv(filename)
    time = df["Time (s)"]
    acc = df["Linear Acceleration y (m/s^2)"]

elif dataType == 2: # 刘楷0步行
    filename = "data/刘楷0步行左手食指指环7E_EE_2022_07_06_1014_3985_ACC.csv"
    df = pd.read_csv(filename)
    # df = pd.read_csv(filename, nrows=1000)
    time = df["timestamp"]
    # acc = df["accel_Y"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)

elif dataType == 3: # 刘楷0慢跑
    filename = "data/刘楷0慢跑左手食指指环7E_EE_2022_07_06_1019_5089_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)

elif dataType == 4: # 刘楷1慢跑
    filename = "data/刘楷1慢跑右手食指指环7E_EE_2022_07_06_1101_4849_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)

elif dataType == 5: # 刘伟杰1
    filename = "data/刘伟杰1步行左手食指指环7E_EE_2022_07_06_1055_4497_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)

elif dataType == 6: # 刘伟杰步行
    filename = "data/刘伟杰步行左手食指指环7E_EE_2022_07_06_1040_4577_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)

elif dataType == 7: # 张建宇步行
    filename = "data/张建宇步行左手食指指环7E_EE_2022_07_06_1030_4497_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)

else:
    print("error")


# 滑动均值滤波
avg_time = get_rolling_avg(time, window_size)
avg_acc = get_rolling_avg(acc, window_size)

# find out if the data trend is increasing
# 找到递增数据的趋势
series_avg_acc = pd.Series(avg_acc) # list转Series

# 当前帧为i，a[i]-a[i-1]>0，是否递增
# diff()前后两帧作差，ge(0)和0比较大小，输出bool
increasing_elements = series_avg_acc.diff().ge(0)

# 当前帧为i，a[i]-a[i+1]<0，是否递减
# shifted trend (local minima) 局部最小值？
# Series右移一位，第一帧为nan。然后和原序列进行异或操作
shifted = increasing_elements.ne(increasing_elements.shift())

# find the local max
# Series可以用~取反，布尔值本身不能~取反
# 这里求的是local_min不是local_max
local_max = shifted & (~increasing_elements)

# generate the step data by grouping sthe acceleration by the threshold
# 通过将加速度按阈值分组来生成步长数据

#step = series_avg_acc.groupby(count_extreme_points).apply(threshold_fn)

count_extreme_points = local_max.cumsum() # 对极值点个数累加
print('type(count_extreme_points): {}'.format(type(count_extreme_points)))
print('count_extreme_points: {}'.format(count_extreme_points))

# series_avg_acc按照count_extreme_points进行分组
grouped = series_avg_acc.groupby(count_extreme_points)

#查看分组
# print(grouped.groups)

step = grouped.apply(threshold_fn)

print('step: {}'.format(step))
# print('step: {}'.format(type(step)))
# print('series_avg_acc.min(): {}'.format(series_avg_acc.min()))

# 找到series_avg_acc最小值，整个Series加上这个值
step += series_avg_acc.min()

# difference from the previous row
step_change = step - step.shift(1) # step_change[i]=step[i]-step[i-1]

# generate the final value count
value_count = step_change.value_counts()

positive_count = value_count[value_count.index > 0].iloc[0] #

# plot the data using matplotlib
fig, ax = plt.subplots()

ax.plot(avg_time, avg_acc, label="The Average Acceleration")
ax.plot(avg_time, step, drawstyle='steps', label="Step")
ax.legend()

# plotting settings
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration $(m/s^2)$')
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(10))

title_style = {
    'verticalalignment': 'baseline',
    'horizontalalignment': "center"
}

text = "Pedometer Data Results (Total: {num} steps)".format(num=positive_count)
plt.style.use('seaborn')
plt.title(label=text, fontdict=title_style)

print("==============================")
print("The Pedometer Algorithm result")
print("Total step:", positive_count)
print("==============================")


plt.show()