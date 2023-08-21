项目功能：分割乳腺癌病灶部位

环境：python3.8  PyTorch 1.9.0

data：存放训练和验证数据，但由于该数据属于医院未公开数据，所以只展示测试用例。 若需要请联系本人。

Dataset: 输入数据预处理。

Loss: 训练过程中所使用的的损失函数。

networks：网络初始化。

U_Net: 项目使用的模型。

utils: 其他功能函数。

demo.py: 运行测试的代码，结果存放于demo_result(服务器不能展示图片)。

main.py: 开始训练

U-Net_54.pt：已训练好的模型权重

validation.py: 模型性能评估指标函数