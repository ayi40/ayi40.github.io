---
title: AWS Service
date: 2024/5/23 20:46:25
categories:
  - [AWS]
---

Some simple knowledge of PyG

<!-- more -->





# IAM- Identity and Access Management

创建用户，分组以及分配权限

## IAM Policies



## IAM MFA



## Access AWS

在AWS（Amazon Web Services）中，CLI、SDK和Management Console是三种不同的方式来管理和与AWS服务进行交互。以下是它们的区别和用途：

+ AWS CLI（Command Line Interface）：一个统一的工具，允许用户通过命令行界面与AWS服务进行交互。**CLI** 更加适合脚本和自动化任务。Protected by access keys

+ AWS SDK（Software Development Kit）： 提供了多种编程语言的库和工具包，使开发人员可以在代码中直接与AWS服务进行交互。**SDK** 适合开发人员在应用程序中集成AWS服务。Protected by access keys

+ AWS Management Console： AWS提供的基于Web的用户界面，用于管理AWS资源。**Management Console** 适合管理和监控任务，提供可视化的管理界面，适合不熟悉代码的人操作。Protected by password and + MFA, generate access keys.

## IAM Roles

与IAM user类似，但是不是给人用的，是给AWS Service用的

E.g. 现在有一个EC2 Instance启动需要用到某些AWS功能，我们需要给它分配IAM Roles管理其权限



# EC2

+ EC2 User Data (runs with the root user)
  + lt is possible to bootstrap our instances using an EC2 Wser data script
  + bootstrapping means launching commands when a machine starts
  + That script is only run once at the instance first start
  + EC2 user data is used to automate boot tasks such as:
    + Installing updates
    + Installing software
    + Downloading common files from the internet
    + Anything you can think of





## EC2



## EBS



## ELB



## ASG