import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import vlc
from matplotlib.animation import FuncAnimation

from dsp import ExpFilter

fig, ax = plt.subplots()

# 设置显示界面的位置
mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+120+110")


# 设置显示的柱状图的个数
bin_nums = 24
# 设置显示的频率范围：(0 ~ frequency_threshold Hz)
frequency_threshold = 2700

# 音乐时长(s)
music_length = 0.0

# 帧时长，即每一帧所需的时长,简而言之就是每隔‘sampling_interval’ 秒刷新一次，
# 所以我们所看到的某一个柱状图的状态就是从当前的一帧分析得出的，
sampling_interval = 0.05

temp = np.tile(0, bin_nums)
# 初始化柱状图
rects = ax.bar(range(1, bin_nums + 1), temp)
# 初始化柱状图中的 横线跌落效果
line, = ax.plot(range(1, bin_nums + 1), temp, '_', color='blue', linewidth=16)

# 使用filter,来使得柱形图中的变化平缓一些
# alpha_decay和alpha_rise,其值需在0~1之间，分别表示下降和上升的反应速度
# 越大越灵敏，如果都设为1，将失去滤波的效果
# 这里选择了0.3 和 0.6，即下降时较慢，而上升较快
filter = ExpFilter(np.tile(0,bin_nums),alpha_decay=0.30, alpha_rise=0.60)

def init():
    global y_max
    ax.set_xlim(0, bin_nums + 1)
    ax.set_ylim(0, y_max)
    ax.set_yticks(())
    ax.set_xlabel("frequency ( * " + str(frequency_threshold / (bin_nums * 2)) + " Hz)")
    ax.set_title("Simple Music Visualizer")
    return rects

music_play_start_time = 0
current_time = 0
music_fft = np.empty(0)
bins = np.empty(0)

def update(frame):
    '''此函数会被FuncAnimation调用，已更新界面'''
    current_time = time.time()

    # 获知当前歌曲的播放进度，以选择此进度下的fft数据
    current_frame = ((current_time - music_play_start_time) //
                     sampling_interval)
    if current_frame == FRAMES - 1:
        plt.close(fig)
        return rects

    # 这里的music_fft是从getBin()获得的
    source = music_fft[int(current_frame)]
    index_max = y_max - (y_max // 80) < source
    source[index_max] = y_max - (y_max // 80)
    # 更新柱状图
    bins = filter.update(source)

    # 更新柱状图中的横线跌落效果
    line_ydata = line.get_ydata()
    line_ydata -= int(y_max // 30)
    line_index = line_ydata  - int(y_max // 100) < bins
    line_ydata[line_index] = bins[line_index] + int(y_max // 100)
    line.set_ydata(line_ydata)

    for rect, h in zip(rects, bins):
        rect.set_height(h)
    # print("frame time ： %s true time : %.2f" % (str((current_frame + 1)
                                                    # * sampling_interval), time.time() - music_play_start_time))

    fig.canvas.flush_events()
    fig.canvas.draw()
    # 图像更新后将保持一帧的时间
    time.sleep(sampling_interval)

def getBin(y,sr,sampling_interval):
    '''此函数将会处理音乐中的每一帧数据'''
    time1 = time.time()

    # 计算每一帧有多少“采样“,用于计算fft
    fft_interval = int(sr * sampling_interval)
    length = fft_interval // 2

    # 这里使用的了frequency_threshold,设置柱状图上所显示的频率范围
    # 原本使得 nums = bin_nums即可，这样所显示的频率将是0 ~ 22.1kHz
    # 但是对一般音乐来说 10 ～ 22.1kHz内的能量是非常低的，反应在柱状图上就是右边大半部分都很低
    # 所以这里我将frequency_threshold设为2.7kHz
    nums = (sr * bin_nums) // frequency_threshold

    batch = length // nums

    result = np.atleast_2d(np.tile(0, bin_nums))
    for i in range(int(music_length // sampling_interval)):
        fft = np.fft.fft(y[fft_interval * i: fft_interval * (i + 1)])

        freqbin = np.array([np.abs(fft[batch * x: batch * (x + 1)]).sum() // sampling_interval
                            for x in range(bin_nums)])
        result = np.vstack([result, freqbin])
    time2 = time.time()
    return result

if __name__ == "__main__":
    # 设置音乐文件的路径
    # audio_path = 'your music path'
    audio_path = librosa.util.example_audio_file()

    y, sr = librosa.load(audio_path, sr=None)
    music_length = len(y) / sr

    print("时长: %g s 采样率: %g kHz " % (music_length, sr/1000))
    p = vlc.MediaPlayer(audio_path)

    music_fft = getBin(y=y, sr=sr, sampling_interval=sampling_interval)
    y_max = music_fft.max() // 3
    FRAMES = music_fft.shape[0]
    ani = FuncAnimation(fig, update, init_func=init, blit=False, interval=0,
                        frames=FRAMES + 1, repeat=False)
    print("begin!")
    p.play()
    # 记录音乐开始播放的时间
    music_play_start_time = time.time()
    plt.show(block=False)
    plt.pause(music_length)
