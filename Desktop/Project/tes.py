# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:02:46 2017

@author: Cheng Zou
"""

for i in range(50):
    x_n[2*i+1]=1
NFFT = 2**14
gain = 5
x1=np.arange(8193)*0.1
x=np.arange(0,np.pi*3,0.1) 

y=np.sin(x)
f, ax = plt.subplots(1, 2, sharey=True)
#f, ax = plt.subplots(2, sharey=True)效果同上
# 此时的ax为一个Axes对象数组，包含两个对象
L = len(x)  
y1 = y* np.hamming(x.shape[0])
f = 5/2 * np.linspace(0,1,NFFT/2.0+1)
Y = np.fft.fft(y1*gain,NFFT)/float(L)  
Y_plot = 2*abs(Y[0:int(NFFT/2.0+1)])
ax[0].set_title('Sharing Y axis')
ax[0].plot(x1, Y_plot)
ax[1].plot(x, y)
plotPath= './sin.jpg' # 图片保存路径
plt.savefig(plotPath)   # 保存图片 