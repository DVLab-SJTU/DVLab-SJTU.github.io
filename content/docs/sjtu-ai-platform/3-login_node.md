---
title: 3-登录节点
weight: 4
---

## 如何登录
- **AI平台（使用V100）** 通过`ssh seesw-sub1@login.hpc.sjtu.edu.cn`登录，输入密码进入节点。
- **思源平台（使用A100）** 通过`ssh seesw-sub1@sylogin.hpc.sjtu.edu.cn`登录，输入密码进入节点。
如果要指定一个登录节点，如AI平台中的login2节点，则命令改为`ssh seesw-sub1@login2.hpc.sjtu.edu.cn`。

这里以AI平台为例：
![example](../imgs/login_node_example_1.png)


## 配置conda环境
- 下载miniconda（python3.8），使用wget方式，相对省事。
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
```
- 安装，直接全部默认即可。最终应该是得到了 `/lustre/home/acct-seesw/seesw-sub1/miniconda3`这个目录。
- 接下来随便你自己创建自己的虚拟环境。如我创建了一个`mmdet`的环境，安装了mmdet，安装了pytorch-cu102-1.9.1等。


## 加载一些软件包
比如cuda，cudnn等，平台上安装了非常多，方便大家使用不同的。为了保险起见，一般推荐使用cuda10.2。
- 首先看有哪些已经内置好的包，执行`module avail`，可以看到有非常多。其中D表示默认，如直接写加载cuda，则会找默认版本加载。需要特定版本就要写详细。
![example](../imgs/login_node_example_2.png)
- 通过`module list`可以查看自己当前加载了哪些包。最开始登录时没加载cuda的，所以执行`cuda -V`查看版本会报错。
- 通过`module load xxx`来加载xxx包，如加载gcc编译的cuda-10.2，则执行`module load cuda/10.2.89-gcc-8.3.0`，如图：
![example](../imgs/login_node_example_3.png)
- 这些软件包在debug或者正式提交任务的时候是需要选择的，如需要加载cuda和cudnn。不然比如gpu版本torch会报错。
- 官方文档：[查看官方对于加载软件包的详细说明](https://docs.hpc.sjtu.edu.cn/app/module.html)

## 申请调试资源
{{< hint danger >}}
还是推荐使用组里的服务器调试，省钱！
{{< /hint >}}

**可用资源：**

通过`sinfo`可以看到很多东西：
![example](../imgs/login_node_example_4.png)
- 红框中**dgx2**表示的是gpu的PARTITION（我也不知道中文该怎么叫，就理解为gpu的分区吧，也就是这个PARTITION下都是gpu的节点），是我们在训练时候要用的PARTITION。
- STATE下的idle，mix等表示的是节点的状态，比如idle表示这节点16张卡都没人用，mix表示一部分卡有人用，一部分没有。
- nodelist下表示的是节点名，这比较重要。我们比如申请1个节点的4卡，他就会自动从某个节点（如vol05）下找到4块卡分给你。我推荐大家只申请1个节点的n块卡，而不要申请2个节点的n块卡。节点间通信就涉及到多机多卡，而且速度也很慢！
- 在思源平台上，gpu对应的PARTITION是a100。

**申请调试**

如想申请一块卡或者两块卡进行调试（比如调试ddp训练啊等等）,再比如想申请一个cpu节点进行长时间的解压缩等耗cpu的任务，则使用下列命令来申请：

``` bash
# 申请GPU资源
# -N 1 -n 1 这句就这样不要改！别问我啥意思，大概就是1个节点1个任务。
# -p dgx2 就是从dgx2这个PARTITION申请资源
# --cpus-per-task就表示每张卡要配几核cpu
# --gres=gpu:1表示申请1块gpu，申请n块就改成n
srun -N 1 -n 1 -p dgx2 --cpus-per-task=6 --gres=gpu:1 --pty /bin/bash

# 申请cpu资源
# 参数同理，此时cpu这个PARTITION没有gpu，所以gres:=gpu:1就去掉了
srun -N 1 -n 1 -p cpu --cpus-per-task=6 --pty /bin/bash
```

结果如图所示，会输入命令后会经过 正在申请-申请到 两个阶段，如果当下没有资源可申请到，就会一直在第一句话等待。申请得到后会自动进入申请到的计算节点（如当下已经变成了seese-sub1@vol08)，无论在计算节点还是登录节点都可以输入`squeue`可看到当下申请到的资源队列。在*登录节点中*可通过`ssh vol08`进入到我们申请的计算节点里。你会发现你配的环境，你的目录都和在登录节点一模一样。
![example](../imgs/login_node_example_5.png)


**结束调试，释放资源**

结束调试马上记得释放资源，不然钱一直在花，啊钱一直花～～
在计算节点内执行`exit`退出计算节点，返回到登录节点，通过`squeue`得知作业的jobid，然后使用`scancel jobid`即可释放。
![example](../imgs/login_node_example_6.png)


## 正式提交作业任务
{{< hint danger >}}
这一步就是都调试好后，要开始正经训练了。
{{< /hint >}}
也很简单，在代码文件夹下只需要增加一个启动脚本可。比如命名为`startup.sh`。现在假如我们的代码在`~/codes/ContrastMask/`下，则把启动脚本也放进去。目录结构如下
```tpl
|- ~/codes/ContrastMask
|------ configs
|------ mmdet
|------ libs
|------ tools
|----------- dist_train.sh
|------ startup.sh
```

脚本内容如下所示即可,注意下边代码块里的`#`不是注释，也就是`#SBATCH xxxxxx`是脚本的一部分：
``` bash
#!/bin/bash

#SBATCH --job-name=huiser_BInst
#SBATCH -p dgx2
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4    
#SBATCH --mail-type=all
#SBATCH --mail-user=wangxuehui@sjtu.edu.cn
#SBATCH --output=logs/MaskFCOS_r50_box_P2_64x64_largesize_%j.out
#SBATCH --error=logs/MaskFCOS_r50_box_P2_64x64_largesize_%j.err

module load cuda/10.2.89-gcc-8.3.0 cudnn
source activate mmdet

sh ./tools/dist_train.sh configs/MaskFCOS_fpn_r50_caffe_1x_coco_p3top7_box.py 4 --deterministic --work-dir=./work_dirs
```

详细解释：
- --job-name=huiser_BInst：是该作业的自定义名称为huiser_BInst，两个作业名字相同与否都可以
- -p dgx2: 是指要使用dgx2这个PARTITION中的节点
- -n 1：别问，就这么写，反正就是申请一个节点
- --nodes=1：同上
- --ntasks-per-node=1：同上
- --cpus-per-task=6：每块卡配6核cpu
- --gres=gpu:4：申请4块gpu，一般要申请不同gou数量改这里就可以
- --mail-type=all：指是否在不同状态通过邮件通知，all表示在任务开始启动，任务结束，任务出错时候都发邮件通知我
- --mail-user=wangxuehui@sjtu.edu.cn：被通知的邮箱
- --output=logs/MaskFCOS_r50_box_P2_64x64_largesize_%j.out： 任务输出日志路径，%j表示jobid
- --error=logs/MaskFCOS_r50_box_P2_64x64_largesize_%j.err：任务错误信息输出路径
- module load cuda/10.2.89-gcc-8.3.0 cudnn：加载cuda-10.2.89的包，加载默认版本的cudnn
- source activate mmdet：激活我创建好的mmdet环境。也说明了数据节点，登录节点，计算节点都使用了一套空间。
- 最后一句就是要执行的代码的命令，写python也好写sh也罢，就像我们在组里的服务器上执行启动命令完全一样了。


**提交任务**

编辑好脚本后，在当前代码文件夹下执行 `sbatch startup.sh`即可，可以看到返回来了jobid，随后可通过`squeue`查看任务状态，如PD是pending等待中，R是running运行中，CG是运行结束完成。任务运行后就可以在代码里写好的输出路径下看到东西了。

