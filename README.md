# fsl_ts
few shot time series predict / anomaly detection based on MAML

## 包依赖
* python3.*
* pandas
* numpy
* matplotlib
* pytorch(torch 1.6.0+cu101)
* higher

## 包依赖安装说明
使用如下命令进行python环境配置
```
pip3 install -r requirements.txt
```

## 程序运行说明
在项目根路径下通过使用脚本来运行程序，运行脚本前需要先赋予执行权限，即输入以下命令：
```
chmod +x meta_run.sh
```
执行脚本
```
'meta_run.sh':运行few shot time series prediction脚本程序
```

脚本中含有较多的超参数，详情可以参考'超参数说明'

## 说明
**代码结构**
```
-- fsl_ts_dataloader.py：用于获取meta train的batch task训练数据
-- fsl_ts_maml.py：few shot learning的主控程序，包含train、evaluate、test、predict等多个过程，且包含超参数含义与默认值
-- lstm_learner.py：lstm模型，MAML中的base model
-- maml_higher_learner：MAML算法的主要实现模块，包含meta_train和fine_tune
-- plot_tools.py：画图工具类
-- meta_run.sh：执行few shot learning的脚本程序
```

**数据说明**

目前使用了4个KPI数据集，已经预处理完毕并置于以下目录：
```
-- fsl_generator
---- fsl_pool
------- xxx_spt_x_pool.npy
------- xxx_spt_y_pool.npy
------- xxx_qry_x_pool.npy
------- xxx_qry_y_pool.npy
------- ......
```
也可以使用自定义数据集与dataloader，输入格式符合MAML模块、lstm模块对support set和query set的要求即可。

**超参数说明**

meta_run.sh中有一些超参数，含义如下：

*路径类参数*
```
model_params_dir: 模型的保存路径
figure_dir: 训练相关图像的输出路径
logs_dir: 日志存放路径
logs_name: 日志名称，默认输出都存放在该日志中，命令行无输出，需要观察输出时可用cat命令捕获该日志内容
check_point: 初始化模型参数对应的检查点名称
```
*meta-training参数*
```
epoch: meta-training训练轮次
n_ways: （暂时没用，分类任务中n-way-k-shot的定义）
k_spt: 每个task中support set的数量
k_qry: 每个task中query set的数量
task_num: meta_train阶段task batch对应的batch大小
task_num_eval: meta_evaluate阶段task batch对应的batch大小
task_num_test: meta_test阶段task batch对应的batch大小
meta_lr: meta_update(outer loop)的学习率
update_lr: inner loop的学习率
update_step: meta_train阶段inner loop的更新次数
update_step_test: meta_test阶段fine_tune的更新次数
clip_val: gradient clipping的参数（避免梯度爆炸）
```
*cuda参数*
```
cuda_no: 使用第几号显卡，对应'CUDA_VISIBLE_DEVICES=no'种的'no'
no_cuda: 如果不使用显卡，则在meta_run.sh的python执行命令中加入'--no_cuda'即可
```