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

# region 子函数功能

# get the rolling average of a pandas' series
# 滑动均值滤波
def get_rolling_avg(input_series, window_size):
    windows = input_series.rolling(window_size)

    # Create a list of moving averages
    moving_averages = windows.mean().tolist() # tolist()数组转换成列表
    return moving_averages[window_size - 1:] # 返回剩余数据，前window_size个值为nan

'''
thresholding function
参数列表：
    x: 一段长度的avg_acc
    threshold：用来筛选
    stepH：用来表示一步的高度
    res: 长度和x相同的0向量，在满足阈值的帧的位置上置为step
作用：
    筛选出输入序列x中，比最小值大threshold的位置，不记录具体大多少。
    并用一个长度相等的向量记录结果
'''

def threshold_fn(x, threshold=7.8, stepH=0.01): # threshold=0.0035 除1000的时候
    # 生成了一个长度为len(x)的0向量
    res = pd.Series(0*len(x), index=x.index)

    t = x.min() + threshold # x.min()就是局部最小值
    bool_meet_thr = x.gt(t) # 用bool向量记录x序列中满足大于t的位置
    res.loc[bool_meet_thr] = stepH # 行用loc操作，每个数据对应的变成了列标签

    # for idx,data in x.items(): # 遍历DataFrame
    #     # print("[{}]: {}".format(idx,data))
    #     if data > threshold:
    #         res[idx] = x[idx]-x.min()
    #         print('res[idx]:', data)

    return res

# endregion

# input file: imported phyphox accelerometer -- without g
# also try: walk-10-step-2022-2-24-v2.csv !
# Note: This will not work with data with g. (The value range is different)

# region 初始参数获取

window_size = 100 # 窗就是帧的个数，根据步伐的频率决定
dataType = 4

# 读取汇总文件
# 从中获取所有的真实步数 todo


# 读取数据文件
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
    real_step_num = 163  # 通过金标准获取的真实步数

elif dataType == 3: # 刘楷0慢跑
    filename = "data/刘楷0慢跑左手食指指环7E_EE_2022_07_06_1019_5089_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)
    real_step_num = 448  # 通过金标准获取的真实步数

elif dataType == 4: # 刘楷1慢跑
    filename = "data/刘楷1慢跑右手食指指环7E_EE_2022_07_06_1101_4849_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)
    real_step_num = 504  # 通过金标准获取的真实步数

elif dataType == 5: # 刘伟杰1
    filename = "data/刘伟杰1步行左手食指指环7E_EE_2022_07_06_1055_4497_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)
    real_step_num = 372  # 通过金标准获取的真实步数

elif dataType == 6: # 刘伟杰步行
    filename = "data/刘伟杰步行左手食指指环7E_EE_2022_07_06_1040_4577_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)
    real_step_num = 180  # 通过金标准获取的真实步数

elif dataType == 7: # 张建宇步行
    filename = "data/张建宇步行左手食指指环7E_EE_2022_07_06_1030_4497_ACC.csv"
    df = pd.read_csv(filename)
    time = df["timestamp"]

    accel_X = df["accel_X"]
    accel_Y = df["accel_Y"]
    accel_Z = df["accel_Z"]
    acc = np.sqrt(accel_X**2+accel_Y**2+accel_Z**2)
    real_step_num = 285  # 通过金标准获取的真实步数

else:
    print("error")

# 测试模式 1进入，0退出
testMode = 0

# endregion


# 滑动均值滤波
avg_time = get_rolling_avg(time, window_size)
avg_acc = get_rolling_avg(acc, window_size)

# find out if the data trend is increasing
# 找到递增数据的趋势
series_avg_acc = pd.Series(avg_acc) # list转Series

# 当前帧为i，a[i]-a[i-1]>0，是否递增
# diff()前后两帧作差，ge(0)返回大于等于0的bool向量
increasing_elements = series_avg_acc.diff().ge(0)

# 当前帧为i，a[i]-a[i+1]<0，是否递减
# shifted trend (local minima) 局部最小值？
lag_one_frame = increasing_elements.shift() # 右移滞后一位，第一帧为nan
shifted = increasing_elements.ne(lag_one_frame) # 和原序列进行异或

# find the local max
# 这里求的是local_min不是local_max
local_max = shifted & (~increasing_elements) # Series用~取反，布尔值本身不能~取反

# generate the step data by grouping the acceleration by the threshold
# 通过将加速度按阈值分组来生成步长数据
# step = series_avg_acc.groupby(local_max.cumsum()).apply(threshold_fn)

accumulated_extremum_nums = local_max.cumsum() # 当前帧累积极值点个数的向量

# series_avg_acc按照accumulated_extremum_nums进行分组
grouped = series_avg_acc.groupby(accumulated_extremum_nums) # 按累积极值点个数分类

# 查看分组
# print(grouped.groups)

step = grouped.apply(threshold_fn)

print('step: {}'.format(step))
# print('step: {}'.format(type(step)))
# print('series_avg_acc.min(): {}'.format(series_avg_acc.min()))

# 找到series_avg_acc最小值，整个Series加上这个值
step += series_avg_acc.min()

# # difference from the previous row
# step_change = step - step.shift(1) # step_change[i]=step[i]-step[i-1]
#
# # generate the final value count
# value_count = step_change.value_counts()
#
# positive_count = value_count[value_count.index > 0].iloc[0] #

# difference from the previous row
# 与前一帧作差，判断当前位置是递增还递减
step_change = step - step.shift(1) # step_change[i]=step[i]-step[i-1]，变化值
# generate the final value count
gradient_count = step_change.value_counts() # 统计不同斜率个数


# index表示增减性，获取递增趋势的位置
positive_count = gradient_count[gradient_count.index > 0]
# 取正向斜率为预测步数
estimate_step_count = positive_count.iloc[0] # 取positive_count第0行数据

# 相对误差 = |测量值-真实值|/真实值
relative_error = abs(estimate_step_count-real_step_num)/real_step_num
# 计算预测准确率
estimate_accuracy = 1 - relative_error

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
print("Total Estimate step:", estimate_step_count)
print("Real step:", real_step_num)
print("Relative Error: {:.2%}".format(relative_error)) # 相对误差
print("Estimate Accuracy: {:.2%}".format(estimate_accuracy)) # 预测准确率
print("==============================")


plt.show()