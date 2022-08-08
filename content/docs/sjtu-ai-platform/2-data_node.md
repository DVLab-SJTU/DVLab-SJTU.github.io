---
title: 2-数据节点
weight: 3
---

## 如何登录
其实一般情况下是不需要登录的，因为存储共享的原因，无论是上传还是下载，路径都可以在日常使用的登录节点获得。所以这里的登录比较鸡肋。但是还是说一下。

- **AI平台（使用V100）** 通过`ssh seesw-sub1@data.hpc.sjtu.edu.cn`登录，输入密码进入节点。
- **思源平台（使用A100）** 通过`ssh seesw-sub1@sydata.hpc.sjtu.edu.cn`登录，输入密码进入节点。

这里以AI平台为例：
![example](../imgs/data_node_example.png)


## 如何上传，以AI平台为例
使用**scp**命令进行上传，校园网内/外网都可以。如我要上传本地`/home/huiserwang/Downloads/resnet50.pth`到平台`HOME`下的`data/pretrained/`中，则命令为：
``` shell
# seesw-sub1为账号名
# 建议传数据都打包好上传，零散文件一大堆比较慢。
scp /home/huiserwang/Downloads/resnet50.pth seesw-sub1@data.hpc.sjtu.edu.cn:~/data/pretrained/
```

传完以后就可以在三种节点中的`~/data/pretrained`目录下看到了。

## 如何下载，以AI平台为例
上述命令反着来就行。

