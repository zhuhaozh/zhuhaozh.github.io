---
layout: post
title:  "ubuntu17.10/18.04 安装cuda小记"
date:   2018-04-25 22:18:00 +0800
categories: ubuntu
tags: pytorch cuda
---
* content
{:toc}

因为使用pytorch，所以打算安装一下cuda，但是中间碰到各种问题目前终于解决，我所使用的系统是ubuntu17.10(原本使用的是ubuntu18.04后来因为显卡问题重装了一下系统)，显卡是intel集显+nivdia940

目前**pytorch最新版本不再支持9系的显卡**，如果需要使用，则必须自己手动编译pytorch





## 过程中遇到的坑
**问题1**. 安装完nvidia驱动后，无法开机（卡在started gnome display manager service...）
这是由于nvidia默认不支持wayland，所以没办法连接到gdm上，需要将gdm换成xorg或通过编译内核让nvidia支持wayland，解决办法在 [这里](https://bugs.launchpad.net/ubuntu/+source/gnome-shell/+bug/1705369/comments/1) 找到，同样的解决办法在[这里](https://charlienewey.github.io/getting-nvidia-drivers-working-on-ubuntu-17-10/)也有提及。

**问题2**. 显卡切换问题
使用 prime-select nvidia 之后黑屏/分辨率异常/循环登录/
黑屏：切换到nivdia之后，重启电脑就再也没法进去，我的笔记本支持在bios中选择使用集显，所以在选择集显之后就可以用集显正常进入（或者进入recovery模式，执行prime-select intel也许也可以，但是我试过不可以），这时出现了另一个问题：分辨率异常

我的电脑上分辨率异常是因为不合适的/etc/X11/xorg.conf导致的，由于执行了nvidia-xconfig，让nivdia默认的配置文件替换掉了，才出现来这样的问题，在重装电脑之后/etc/X11目录下出现了xorg.conf-04252018的文件（不知道为什么？）通过将该文件重命名为xorg.conf之后，分辨率正常，**不要使用nvidia-xconfig来生成新的xorg.conf**

重装系统后，可以正常切换显卡，但是切换到nvidia之后，会导致循环登录，查/var/log/syslog后发现(EE)Failed to initialize GLX extention(compatible nvidia x driver not found)，解决办法也是通过上述更换xorg.conf解决

不知道哪冒出来的这个配置文件，但确实解决了我的问题，**不保证每个电脑上都可以用下面的配置文件解决**，配置文件中的内容如下：
```
Section "ServerLayout"
    Identifier "layout"
    Screen 0 "nvidia"
    Inactive "intel"
EndSection

Section "Device"
    Identifier "intel"
    Driver "modesetting"
    BusID "PCI:0@0:2:0"
    Option "AccelMethod" "None"
EndSection

Section "Screen"
    Identifier "intel"
    Device "intel"
EndSection

Section "Device"
    Identifier "nvidia"
    Driver "nvidia"
    BusID "PCI:1@0:0:0"
    Option "ConstrainCursor" "off"
EndSection

Section "Screen"
    Identifier "nvidia"
    Device "nvidia"
    Option "AllowEmptyInitialConfiguration" "on"
    Option "IgnoreDisplayDevices" "CRT"
EndSection
```

--------

## 步骤
### 1. 是否支持cuda
这步不多说，大家应该都有这个判断能力吧？

### 2. 安装nvidia驱动
根据[该网址](https://charlienewey.github.io/getting-nvidia-drivers-working-on-ubuntu-17-10/)的步骤安装显卡驱动
具体步骤如下：
#### (1) 删除已有的nvidia驱动
执行以下指令，系统将自动删除以'nvidia'开头的软件;
sudo apt-get autoremove --purge '^nvidia'.
#### (2) 将开源的nouveau驱动设置到黑名单中
由于nvidia可以已经提前替我们做过这一步，所以使用：ls /etc/modprobe.d/nvidia-*.conf 查看是否已经存在相应的文件，同时文件中包含以下的配置内容，如果存在的话，这一步可以跳过。
如果上面提到的文件不存在，则创建一个新文件， /etc/modprobe.d/blacklist-nouveau.conf 并将下面这几行写进去;
```
blacklist nouveau
blacklist lbm-nouveau
alias nouveau off
alias lbm-nouveau off
```
然后执行sudo rmmod nouveau, 再执行 sudo update-initramfs -u
#### (3) 重新安装NVIDIA驱动
添加源：sudo add-apt-repository ppa:graphics-drivers/ppa
目前暂时不要重启，重新安装驱动：sudo apt install nvidia-396（或其他版本的驱动，但cuda9.0以后需要的驱动版本需要384及其以上）

#### (4) 解决上面提到的坑
1. 让nvidia支持wayland：
创建一个新文件：/etc/modprobe.d/nvidia-drm-nomodeset.conf，并写入：options nvidia-drm modeset=1
再次更新执行：sudo update-initramfs -u
此时若重启后，nvidia应该已经可以支持wayland了

2. 为了保险起见，可继续将gdm改为传统的x11
打开/etc/gdm3/custom.conf文件，并取消WaylandEnable=false这一行的注释
此时该配置文件的前几行类似如下

```
# GDM configuration storage
# # See /usr/share/gdm/gdm.schemas for a list of available options.

[daemon] # Uncoment the line below to force the login screen to use Xorg
WaylandEnable=false
```

之后在登录的时候，点击登录旁边的齿轮，选择Ubuntu on xorg(ubuntu 17.10默认使用的是wayland，但ubuntu18.04已经改回默认xorg)

此时通过 nvidia-smi 判断nvidia驱动是否正确安装
如果有类似的输出则表示驱动已经安装成功
![nvidia-sml](/home/zhuhao/Pictures/Screenshot from 2018-04-25 21-53-10.png)

### 3. 安装cuda
由于我们已经自己安装了nvidia的驱动，所以可以使用cuda的runfile形式安装，安装步骤参考[AskUbuntu的提问](https://askubuntu.com/questions/967332/how-can-i-install-cuda-9-on-ubuntu-17-10?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
，我安装的是cuda9.1，类似的版本应该大同小异

#### 为安装cuda做准备
安装后面使用cuda的软件：
``` bash
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```

由于默认的gcc版本是7.2（Ubuntu17.10），而cuda9需要的是gcc-6
所以安装gcc-6/g++-6
``` bash
sudo apt install gcc-6
sudo apt install g++-6
```
目前，必要的软件就准备好了

#### 安装 CUDA 9 + SDK
[在nivdia官方网页](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=runfilelocal)
选择对应的系统版本，并选择安装方式为runfile(local)的形式，下载cuda的runfile文件
或者直接用[下载连接](https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux)下载cuda9.1
并将该文件设置为可执行文件，**注意后面的--override需要加上！**
：
```
chmod +x cuda_xxxxxxx.run
sudo ./cuda_xxxxxxx.run --override
```
之后会出现一些Y/N的选择，由于已经安装好了nvidia驱动，所以我们不需要它来替我们安装驱动，具体的选择如下
```
You are attempting to install on an unsupported configuration. Do you wish to continue?
y
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81?
n
Install the CUDA 9.0 Toolkit?
y
Enter Toolkit Location
[default location]
Do you want to install a symbolic link at /usr/local/cuda?
y
Install the CUDA 9.0 Samples?
y
Enter CUDA Samples Location
[default location]
```

当显示以下信息时，表示安装成功：
```
Driver:   Not Selected

Toolkit:  Installed in /usr/local/cuda-9.1
Samples:  Installed in /home/zhuhao, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-9.1/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-9.1/lib64, or, add /usr/local/cuda-9.1/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-9.1/bin

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-9.1/doc/pdf for detailed information on setting up CUDA.

***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 384.00 is required for CUDA 9.1 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run -silent -driver

Logfile is /tmp/cuda_install_12126.log
```
#### 设置gcc/g++链接
```
sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
```
目的地址具体情况具体分析，需要保持
后面的目的地址是 上面设置的symbolic link位置（默认的可能是/usr/local/cuda-9.1）

#### 测试cuda是否安装成功
```
cd ~/NVIDIA_CUDA-9.0_Samples/5_Simulations/smokeParticles
make
../../bin/x86_64/linux/release/smokeParticles
```
如果出现了一个动态烟雾的图案，则说明成功


### 4. 编译安装pytorch


## 总结的教训
1. 注意看日志文件
2. 控制变量法
3. 多用google
