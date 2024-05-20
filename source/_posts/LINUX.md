---
title: LINUX
date: 2024/4/10 20:46:25
categories:
  - [Basic Of Computer]
---



<!-- more -->

# 一些常用方法

## 添加环境变量

使用vim编辑环境变量文件（只修改当前用户的环境变量）

```
$ vim ~/.bashrc
```

将下面的命令加入

```
$ export PATH=$PATH:/usr/local/xxxxxx
```

最后，用source在当前bash环境下读取并执行FileName中的命令。

```
$ source ~/.bashrc
```



Note:

Hadoop环境变量配置cdoe如下：

```
export HADOOP_HOME=/安装目录
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```

