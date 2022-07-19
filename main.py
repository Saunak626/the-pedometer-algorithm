import pandas as pd
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

import warnings
from pandas.core.common import SettingWithCopyWarning

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号



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
    moving_averages = windows.mean().tolist()  # tolist()数组转换成列表
    return moving_averages[window_size - 1:]  # 返回剩余数据，前window_size个值为nan


"""
我们的解决方案要求我们对时间序列进行多次过滤。
与其在整个程序中添加过滤代码，不如创建一个负责过滤的类，如果我们需要加强或修改它，
我们只需要改变这一个类就可以了。
"""


class Filter:
    @staticmethod
    def low_0_hz(data):  # 让0Hz附近频率的信号通过
        COEFFICIENTS_LOW_0_HZ = {
            'alpha': [1, -1.979133761292768, 0.979521463540373],
            'beta':  [0.000086384997973502, 0.000172769995947004, 0.000086384997973502]
        }
        return Filter.filter(data, COEFFICIENTS_LOW_0_HZ)

    @staticmethod
    def low_5_hz(data):  # 让低于5Hz频率的信号通过
        COEFFICIENTS_LOW_5_HZ = {
            'alpha': [1, -1.80898117793047, 0.827224480562408],
            'beta':  [0.095465967120306, -0.172688631608676, 0.095465967120306]
        }
        return Filter.filter(data, COEFFICIENTS_LOW_5_HZ)

    @staticmethod
    def high_1_hz(data):  # 让高于1Hz频率的信号通过
        COEFFICIENTS_HIGH_1_HZ = {
            'alpha': [1, -1.905384612118461, 0.910092542787947],
            'beta':  [0.953986986993339, -1.907503180919730, 0.953986986993339]
        }
        return Filter.filter(data, COEFFICIENTS_HIGH_1_HZ)

    @staticmethod
    def filter(data, coef):
        fd = np.zeros_like(data)  # filtered_data

        # IIR数字滤波器公式实现
        for i in range(2, len(data)):
            fd[i] = coef['alpha'][0] * (data[i] * coef['beta'][0]
                                        + data[i - 1] * coef['beta'][1]
                                        + data[i - 2] * coef['beta'][2]
                                        - fd[i - 1] * coef['alpha'][1]
                                        - fd[i - 2] * coef['alpha'][2])
        return fd


'''
画三轴加速度的图像
plot three-axis acceleration data
    data3D: DataFrame
    ax:
    
    return:
'''


def plot_3axis_accel(time, acc_3axis, ax=None):
    x = acc_3axis["accel_X"]
    y = acc_3axis["accel_Y"]
    z = acc_3axis["accel_Z"]

    if not ax:
        plt.figure(dpi=100, figsize=(10, 5))
        ax = plt

    ax.plot(time, x, label='x')
    ax.plot(time, y, label='y')
    ax.plot(time, z, label='z')
    ax.legend(loc='upper right')  # 定义图标所处位置，这里表示右上
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.8)


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


# 7.8 合加速度
# 0.0008 重力分量
def threshold_fn(x, threshold=20, stepH=10):  # threshold=0.0035 除1000的时候
    # 生成了一个长度为len(x)的0向量
    res = pd.Series(0 * len(x), index=x.index)

    t = x.min() + threshold  # x.min()就是局部最小值
    bool_meet_thr = x.gt(t)  # 用bool向量记录x序列中满足大于t的位置
    res.loc[bool_meet_thr] = stepH  # 行用loc操作，每个数据对应的变成了列标签

    # for idx,data in x.items(): # 遍历DataFrame
    #     # print("[{}]: {}".format(idx,data))
    #     if data > threshold:
    #         res[idx] = x[idx]-x.min()
    #         print('res[idx]:', data)

    return res


# endregion
# 关闭警告
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
plt.close('all')  # 关闭脚本中所有打开的图形
sampling_interval = 40  # 每一帧采样之间间隔的时间，40ms
# 窗就是帧的个数，根据步伐的频率决定
window_size = 50  # 40ms * 100 = 4s

# region 初始参数获取

# 读取汇总文件
# 从中获取所有的真实步数 todo

# 读取路径下所有文件名
path = "D:/Github/the-pedometer-algorithm/data"  # todo 按天修改文件路径
data_names = os.listdir(path)
# for i in data_names:
#     print(i)

# 我们的三轴原始的度数accel_Y为实际重力方向，且方向相反

# 读取数据文件

dataTime = 7_8  # 数据测试日期 7_7; 7_6
dataType = 4
if dataTime == 7_7:
    if dataType == 1:  # 刘楷步行
        filename = "data/2022_07_07/1刘楷步行左手食指指环7E_EE_2022_07_07_1556_2113_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]
        real_step_num = 86  # 通过金标准获取的真实步数

    elif dataType == 2:  # 刘楷步行
        filename = "data/2022_07_07/2刘楷步行左手食指指环7E_EE_2022_07_07_1614_1809_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 113  # 通过金标准获取的真实步数

    elif dataType == 3:  # 刘楷步行
        filename = "data/2022_07_07/3刘楷步行左手中指指环7E_EE_2022_07_07_1622_1745_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 115  # 通过金标准获取的真实步数

    elif dataType == 4:  # 刘楷步行
        filename = "data/2022_07_07/4刘楷步行左手中指指环7E_EE_2022_07_07_1625_1761_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 111  # 通过金标准获取的真实步数

    elif dataType == 5:  # 刘楷步行
        filename = "data/2022_07_07/5刘楷步行右手食指指环7E_EE_2022_07_07_1629_1825_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 99  # 通过金标准获取的真实步数

    elif dataType == 6:  # 刘楷慢跑
        filename = "data/2022_07_07/6刘楷慢跑右手食指指环7E_EE_2022_07_07_1635_1953_ACC.csv"
        df = pd.read_csv(filename, nrows=500)
        # df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 147  # 通过金标准获取的真实步数

    else:
        print("error")

    # endregion

elif dataTime == 7_6:
    # region 加速度数据读取
    if dataType == 1:  # 自带数据
        filename = "data/walk-10-step-2022-3-2-v1.csv"
        df = pd.read_csv(filename)
        time = df["Time (s)"]
        acc_total = df["Linear Acceleration y (m/s^2)"]

    elif dataType == 2:  # 刘楷0步行
        filename = "data/2022_07_06/刘楷0步行左手食指指环7E_EE_2022_07_06_1014_3985_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 163  # 通过金标准获取的真实步数

    elif dataType == 3:  # 刘楷0慢跑
        filename = "data/2022_07_06/刘楷0慢跑左手食指指环7E_EE_2022_07_06_1019_5089_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 448  # 通过金标准获取的真实步数

    elif dataType == 4:  # 刘楷1慢跑
        filename = "data/2022_07_06/刘楷1慢跑右手食指指环7E_EE_2022_07_06_1101_4849_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 504  # 通过金标准获取的真实步数

    elif dataType == 5:  # 刘伟杰1
        filename = "data/2022_07_06/刘伟杰1步行左手食指指环7E_EE_2022_07_06_1055_4497_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 372  # 通过金标准获取的真实步数

    elif dataType == 6:  # 刘伟杰步行
        filename = "data/2022_07_06/刘伟杰步行左手食指指环7E_EE_2022_07_06_1040_4577_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 180  # 通过金标准获取的真实步数

    elif dataType == 7:  # 张建宇步行
        filename = "data/2022_07_06/张建宇步行左手食指指环7E_EE_2022_07_06_1030_4497_ACC.csv"
        df = pd.read_csv(filename, nrows=500)
        time = df["timestamp"]

        real_step_num = 285  # 通过金标准获取的真实步数

        # raw_accel_X = df["accel_X"]  # Series
        # raw_accel_Y = df["accel_Y"]
        # raw_accel_Z = df["accel_Z"]
        # Series合并成DataFrame
        # acc_3axis = pd.DataFrame(raw_accel_X, raw_accel_Y, raw_accel_Z)

elif dataTime == 7_8:
    # region 加速度数据读取
    if dataType == 1:  # 自带数据
        filename = "data/2022_07_08/1刘楷步行左手食指指环7E_EE_2022_07_08_1519_5217_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 292  # 通过金标准获取的真实步数

    elif dataType == 2:  # 刘楷0步行
        filename = "data/2022_07_08/2刘楷步行左手食指指环7E_EE_2022_07_08_1526_5169_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 353  # 通过金标准获取的真实步数

    elif dataType == 3:  # 刘楷0慢跑
        filename = "data/2022_07_08/3刘楷步行右手食指指环7E_EE_2022_07_08_1638_5425_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 324  # 通过金标准获取的真实步数

    elif dataType == 4:  # 刘楷1慢跑
        filename = "data/2022_07_08/4刘楷慢跑右手食指指环7E_EE_2022_07_08_1644_5089_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 453  # 通过金标准获取的真实步数

    elif dataType == 5:  # 刘伟杰1
        filename = "data/2022_07_08/5刘楷慢跑左手中指指环7E_EE_2022_07_08_1721_5569_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 478  # 通过金标准获取的真实步数

    elif dataType == 6:  # 刘伟杰步行
        filename = "data/2022_07_08/6刘楷慢跑左手食指指环7E_EE_2022_07_08_1728_5249_ACC.csv"
        df = pd.read_csv(filename)
        time = df["timestamp"]

        real_step_num = 492  # 通过金标准获取的真实步数

else:
    print("error")

# endregion


# 加速度数据处理
acc_3axis = df[["accel_X", "accel_Y", "accel_Z"]]  # DataFrame

# 单位转换 1000
acc_3axis["accel_X"] = acc_3axis["accel_X"] / 1000
acc_3axis["accel_Y"] = acc_3axis["accel_Y"] / 1000
acc_3axis["accel_Z"] = acc_3axis["accel_Z"] / 1000

# 加速度的平方根
df["accel_sum_squares"] = df["accel_X"] ** 2 + df["accel_Y"] ** 2 + df["accel_Z"] ** 2
acc_total = df["accel_sum_squares"].apply(np.sqrt)

# 去趋势
# acc_total_np = np.array(acc_total).reshape(-1, 1)
# acc_total_de = signal.detrend(acc_total_np, axis=0, type='linear')  # 'constant' 信号中的每个数据减去平均值，去除直流
# acc_total_de2 = acc_total_de.reshape(-1)  # 将行向量转化为array数组
# acc_total = pd.Series(acc_total_de2)
# print(acc_total)

# Time offset correction 矫正时间起始偏移
time = (time - time[0]) / 1000  # 时间戳是以ms为单位，转换为s

# 测试模式 1进入，0退出
testMode = 0

total_accel_mode = 1  # 开启合加速度模式

# 滑动均值滤波
filter_mode = 1  # 0: 无滤波; 1: 均值滤波,
if filter_mode == 0:
    avg_acc = acc_total  # 从数据集读取
    avg_time = time  # 从数据集读取
elif filter_mode == 1:
    avg_time = get_rolling_avg(time, window_size)
    avg_acc = get_rolling_avg(acc_total, window_size)

    # 均值滤波
    filtered = []
    df_filtered = pd.DataFrame(filtered)
    df_filtered["accel_X"] = get_rolling_avg(acc_3axis["accel_X"], window_size)
    df_filtered["accel_Y"] = get_rolling_avg(acc_3axis["accel_Y"], window_size)
    df_filtered["accel_Z"] = get_rolling_avg(acc_3axis["accel_Z"], window_size)

# 时间数据长度 testMode
if testMode == 1:
    print('len(time)', len(time))
    print('len(acc_total)', len(acc_total))
    print('len(avg_time)', len(avg_time))
    print('len(avg_acc) ', len(avg_acc))

avg_time = pd.Series(avg_time)  # list转Series
series_avg_acc = pd.Series(avg_acc)  # list转Series

# print('acc_3axis: ', acc_3axis.axes) # 查看df表头信息

plot_3axis_accel(time, acc_3axis)
plt.title('原始xyz加速度')

plt.figure(dpi=100, figsize=(10, 5))
plt.plot(time, acc_total, label='原始合加速度')
plt.plot(avg_time, avg_acc, label='均值滤波后合加速度')
plt.title('合加速度')

# print('type(acc_3axis):{}', type(acc_3axis))
# print('type(np.array(acc_3axis)):{}', type(np.array(acc_3axis)))
# print('np.array(acc_3axis):{}', np.array(acc_3axis))

# 低通滤波器0.3Hz设计的效果不对。并且没确定是在均值滤波前还是后
# data_gravity = Filter.low_0_hz(np.array(acc_3axis))  # data_gravity为重力在芯片的三轴上的分量
# column_names = ["accel_X", "accel_Y", "accel_Z"]
# data_gravity = pd.DataFrame(data_gravity, columns=column_names)
# plot_3axis_accel(time, data_gravity)
# plt.title('1 gravity acceleration')


# sos = signal.butter(10, 0.3, 'lp', fs=4000, output='sos')
# filtered = signal.sosfilt(sos, np.array(acc_3axis))
# print(type(filtered))
# column_names = ["accel_X", "accel_Y", "accel_Z"]
# filtered = pd.DataFrame(filtered, columns=column_names)
# plot_3axis_accel(time, filtered)
# plt.title('2 gravity acceleration')

# sos = signal.butter(10, 0.3, 'lp', fs=4000, output='sos')
# filtered = signal.sosfilt(sos, np.array(acc_3axis))

if total_accel_mode == 0:
    # region

    sample_fre = 100
    nyquist_frequency = sample_fre * 0.5
    cutoff = 3 / 10  # 截止频率 f=1/T 0.3Hz低通滤波
    lowcut = cutoff / nyquist_frequency
    b, a = signal.butter(2, lowcut, 'lowpass')

    low_filtered = []
    df_low_filtered = pd.DataFrame(low_filtered)

    df_low_filtered["accel_X"] = signal.filtfilt(b, a, df_filtered["accel_X"])
    df_low_filtered["accel_Y"] = signal.filtfilt(b, a, df_filtered["accel_Y"])
    df_low_filtered["accel_Z"] = signal.filtfilt(b, a, df_filtered["accel_Z"])

    cutoff = 50 / 10  # 截止频率 f=1/T 5Hz低通滤波
    lowcut = cutoff / nyquist_frequency
    b, a = signal.butter(2, lowcut, 'lowpass')
    df_filtered["accel_X"] = signal.filtfilt(b, a, df_filtered["accel_X"])
    df_filtered["accel_Y"] = signal.filtfilt(b, a, df_filtered["accel_Y"])
    df_filtered["accel_Z"] = signal.filtfilt(b, a, df_filtered["accel_Z"])

    # 计算加速度的平方根
    df_low_filtered["accel_sum_squares"] = df_low_filtered["accel_X"] ** 2 + df_low_filtered["accel_Y"] ** 2 + \
                                           df_low_filtered["accel_Z"] ** 2
    acc_total_fil = df_low_filtered["accel_sum_squares"].apply(np.sqrt)

    plot_3axis_accel(avg_time, df_low_filtered)
    plt.plot(avg_time, acc_total_fil, label='重力')
    plt.legend(loc='upper right')
    plt.title('gravity acceleration')

    # 分离出的用户加速度
    data_user = df_filtered - df_low_filtered
    plot_3axis_accel(avg_time, data_user)
    plt.title('user acceleration')

    # 对比原始加速度和分离出的用户加速度
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plot_3axis_accel(avg_time, df_filtered, axs[0])
    plot_3axis_accel(avg_time, data_user, axs[1])
    axs[0].set_title('total acceleration')
    axs[1].set_title('user acceleration')

    # Dot Product
    '''
    因为numpy.dot不支持多组二维向量内积，我们先用对应元素相乘，
    再把每一行数据加起来就可以得到每一行的点乘结果：
    '''
    data_a = np.sum(data_user * df_filtered, axis=1)
    series_avg_acc = data_a
    plt.figure()
    plt.plot(data_a)
    plt.title("Dot Product Result")

    # endregion

# Noise reduction 过滤掉高频和低频噪音
#
# data_filtered = Filter.low_5_hz(data_a)
# data_filtered = Filter.high_1_hz(data_filtered)
# plt.figure()
# plt.plot(data_filtered)
# plt.ylim([-2, 2])
# plt.title("过滤掉高频和低频噪音")


# region 频域分析模块
#
# Fs = 100  # 采样频率
# T = 1 / Fs  # 采样周期，只相邻两数据点的时间间隔
# L = len(avg_time)  # 信号长度
#
# # 计算信号的傅里叶变换
# Y = fft(acc_total.tolist())  # avg_acc为ndarray
# p2 = np.abs(Y)  # 双侧频谱
# p1 = p2[:int(L/2)]
#
# # 定义频域f并绘制单侧幅值频谱P1。与预期相符，由于增加了噪声，幅值并不精确等于 0.7 和 1。
# f = np.arange(int(L/2))*Fs/L
# plt.figure()
# plt.plot(f, 2*p1/L)
# plt.xlim(-1, 2)  # 坐标轴范围
# plt.title('Single-Sided Amplitude Spectrum of X(t)')
# plt.xlabel('f (Hz)')
# plt.ylabel('|P1(f)|')
#
# # 峰值检测
# P1 = 2 * p1 / L
# plt.figure()
# peaks, _ = signal.find_peaks(P1, height=0)  # 返回找到峰值的值的索引
# print('P1[peaks]:', P1[peaks])
#
# plt.plot(peaks, P1[peaks])
# plt.title("峰值检测")
#
# # endregion


# find out if the data trend is increasing
# 找到递增数据的趋势

# 当前帧为i，a[i]-a[i-1]>0，是否递增
# diff()前后两帧作差，ge(0)返回大于等于0的bool向量


increasing_elements = series_avg_acc.diff().ge(0)

# 当前帧为i，a[i]-a[i+1]<0，是否递减
# shifted trend (local minima) 局部最小值
lag_one_frame = increasing_elements.shift()  # 右移滞后一位，第一帧为nan
shifted = increasing_elements.ne(lag_one_frame)  # 和原序列进行异或

# find the local_min
local_min = shifted & (~increasing_elements)  # Series用~取反，布尔值本身不能~取反

# 增减判断bool向量 testMode
if testMode == 1:
    print('increasing_elements: ', increasing_elements)
    print('lag_one_frame ', lag_one_frame)
    print('shifted : ', shifted)
    print('~increasing_elements: ', ~increasing_elements)
    print('local_min : ', local_min)

# generate the step data by grouping the acceleration by the threshold
# 通过将加速度按阈值分组来生成步长数据
accumulated_extremum_nums = local_min.cumsum()  # 当前帧累积极值点个数的向量

# series_avg_acc按照accumulated_extremum_nums进行分组
grouped = series_avg_acc.groupby(accumulated_extremum_nums)  # 按累积极值点个数分类

# 查看DataFrame分组 testMode
if testMode == 1:
    print('----list(accumulated_extremum_nums): ', list(accumulated_extremum_nums))
    print('----list(grouped.groups)--输出DataFrame的组名-------')
    print(list(grouped.groups))
    print('----list(grouped.groups)--输出子DataFrame的组成员---')
    print(grouped.groups)
    print('----list(grouped)---------输出子DataFrame-----------')
    print(list(grouped))

step = grouped.apply(threshold_fn)  # 可能的步伐
step += series_avg_acc.min()  # 阶跃图基础高度

# difference from the previous row
# 与前一帧作差，判断当前位置是递增还递减
step_change = step - step.shift(1)  # step_change[i]=step[i]-step[i-1]，变化值
# generate the final value count
gradient_count = step_change.value_counts()  # 统计不同斜率个数

# 步伐变化list testMode
if testMode == 1:
    print('----list(step): ', list(step))
    print('----list(step_change):', list(step_change))
    print('----gradient_count:', gradient_count)

# index表示增减性，获取递增趋势的位置
positive_count = gradient_count[gradient_count.index > 0]
# 取正向斜率为预测步数
estimate_step_count = positive_count.iloc[0]  # 取positive_count第0行数据

# 相对误差 = |测量值-真实值|/真实值
# relative_error = abs(estimate_step_count - real_step_num) / real_step_num

relative_error = (estimate_step_count - real_step_num) / real_step_num
# 计算预测准确率
estimate_accuracy = 1 - relative_error

# plot the data using matplotlib
fig, ax = plt.subplots(dpi=100, figsize=(10, 5))
ax.plot(avg_time, avg_acc, label="The Average Acceleration")
ax.plot(avg_time, step, drawstyle='steps', label="Step")
ax.legend()

# plotting settings
ax.set_xlabel('Time (s)')
ax.set_ylabel('Acceleration $(m/s^2)$')
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.xaxis.set_major_locator(MaxNLocator(10))

title_style = {
    'verticalalignment':   'baseline',
    'horizontalalignment': "center"
}

text = "Pedometer Data Results (Total: {} steps)".format(estimate_step_count)
plt.style.use('seaborn')
plt.title(label=text, fontdict=title_style)

print("===============================")
print("The Pedometer Algorithm result")
print("dataTime:{}, dataType:{}".format(dataTime, dataType))
print("total_accel_mode:{}, filter_mode:{}, window_size:{}".format(total_accel_mode, filter_mode, window_size))
print("Total Estimate step:", estimate_step_count)
print("Real step:", real_step_num)
print("Relative Error: {:.2%}".format(relative_error))  # 相对误差
print("Estimate Accuracy: {:.2%}".format(estimate_accuracy))  # 预测准确率
print("===============================")

plt.show()
