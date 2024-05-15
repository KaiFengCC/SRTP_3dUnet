## **医学问题分析** 

​		癫痫是一种由大脑异常放电引起的慢性神经系统疾病，常表现为反复发作的抽搐和意

识障碍。大脑作为人体最复杂的器官之一，拥有数十亿个神经细胞，这些细胞通过复杂的

电化学信号网络来维持正常的脑功能。当这些信号网络出现异常时，便可能引发癫痫发

作，进而影响患者的生活质量。癫痫的发病机制复杂，涉及多种神经元活动的异常和结构

性改变。通过磁共振成像（MRI），可以观察到癫痫患者脑部的结构性变化，如颞叶硬化、

海马体萎缩和皮质发育不良等。这些结构性改变对于癫痫的诊断和治疗具有重要意义。然

而，由于 MRI 图像的复杂性，手动分析这些图像不仅耗时费力，还容易出现误诊或漏诊的

情况。

​		MRI 是一种无创的医学成像技术，具有优异的软组织分辨能力，能够提供高分辨率的

脑部图像，是癫痫诊断的重要工具。通过对癫痫患者的脑部 MRI 图像进行详细分析，医生

可以发现脑部的异常区域，从而确定癫痫的病灶位置。然而，传统的图像分析方法依赖于

医生的经验，容易受到主观因素和疲劳的影响，可能导致诊断准确性降低。

随着计算机技术和人工智能的发展，计算机辅助诊断（CAD）在神经疾病领域的应用越

来越广泛。利用深度学习和机器学习技术，可以对脑部 MRI 图像进行自动分割和分类，提

高诊断的准确性和效率。在癫痫研究中，自动分割技术可以帮助识别癫痫相关的脑区异

常，为医生提供可靠的参考。

​		在本研究中，我们的目标是开发一个基于深度学习的自动分割模型，对癫痫患者的脑

部 MRI 图像进行分析。该模型将能够识别和分割癫痫相关的海马区，通过自动化的图像分

析，可以帮助医生更快、更准确地进行癫痫病灶的定位和评估，减少人为因素造成的误诊

风险。这不仅可以提高诊断的准确性，还能节省医生的时间，使他们能够更专注于患者的

治疗和护理



## **模型选择和调优** 

### **模型选择**

​		**3D UNet** 是一种深度学习模型，专为三维医学图像分割设计。该模型在结构上采用编码

器解码器架构，通过跳跃连接保留高分辨率特征信息，使其在分割任务中表现出色。3D UNet

能够处理三维 MRI 图像，自动识别并分割癫痫相关的异常脑区。

### **3D UNet 模型的优势：**

- 高精度：3D UNet 通过结合局部和全局特征，实现高精度的图像分割。

- 全自动化：减少人为干预，降低主观因素影响，提高诊断一致性。

- 效率高：自动化处理大批量图像，节省医生阅片时间，使其能够专注于患者的治疗和护

理。

- 鲁棒性强：适用于不同的 MRI 图像数据，具有良好的泛化能力。

![image-20240515235340688](C:\Users\fk\AppData\Roaming\Typora\typora-user-images\image-20240515235340688.png)

如图 1，unet 无论是 2D，还是 3D，从整体结构上进行划分，大体可以分位以下两个阶

段：

下采样的阶段，也就是 U 的左边（encoder），负责对特征提取；

上采样的阶段，也就是 U 的右边（decoder），负责对预测恢复。

其中：

蓝色框表示的是特征图；

绿色长箭头，是 concat 操作；

橘色三角，是 conv+bn+relu 的组合；

红色的向下箭头，是 max pool；

黄色的向上箭头，是 up conv；

最后的紫色三角，是 conv，恢复了最终的输出特征图；



### **数据预处理** 

​	在训练模型之前，我们也进行了很多数据预处理工作，以便模型取得最好的学习效果（如

图 2）。

![image-20240515235415806](C:\Users\fk\AppData\Roaming\Typora\typora-user-images\image-20240515235415806.png)

​		sort_nii_image(filePath, imagePath, maskPath)：调用了一个名为

sort_nii_image 的函数，该函数将原始数据文件整理到指定的图像文件夹 imagePath 和掩

码文件夹 maskPath 中。这个函数包括读取原始数据文件、提取图像和掩码等操作。

​		validationDataset = ValidationImageDataSet(dataPath)：创建了一个验证数据集

ValidationImageDataSet 的实例，该数据集包含了从预处理后的图像文件夹和掩码文件夹

中加载数据，并进行必要的转换和预处理操作，例如归一化、裁剪、缩放等。这个过程也

属于数据预处理的一部分。

### **模型构建** 

![image-20240515235437583](C:\Users\fk\AppData\Roaming\Typora\typora-user-images\image-20240515235437583.png)

​		convBlock 类定义了一个卷积块，包括一系列的卷积层、批归一化、Dropout 和激活函

数 LeakyReLU。这个块用于提取输入特征的空间信息，并通过非线性激活函数增强网络的

表达能力。

![image-20240515235453402](C:\Users\fk\AppData\Roaming\Typora\typora-user-images\image-20240515235453402.png)

​		downSample 类定义了一个下采样模块，用于降低特征图的空间分辨率。它包括一个卷

积层和批归一化，通过减少特征图的尺寸来提取更高级别的特征。

![image-20240515235511618](C:\Users\fk\AppData\Roaming\Typora\typora-user-images\image-20240515235511618.png)

upSample 类定义了一个上采样模块，用于增加特征图的空间分辨率。它通过上采样和

卷积操作来融合低级别和高级别的特征，从而实现更精细的分割结果。
