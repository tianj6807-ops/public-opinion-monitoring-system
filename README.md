# 📊 网络舆情监测系统

> 基于 Flask + 朴素贝叶斯 + TextCNN + BERT 的多模型融合网络舆情监测 Web 系统

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📝 项目简介

本项目是一个完整的网络舆情监测 Web 系统，综合运用**传统机器学习**（朴素贝叶斯）、**深度学习**（TextCNN）和**预训练语言模型**（BERT），实现了情感研判、观点群体挖掘和风险预警等功能。

系统采用 **Flask** 搭建后端服务，**ECharts** 实现前端可视化，通过**多模型融合投票机制**提升情感分类的准确性和鲁棒性，最终融合准确率达到 **90.64%**。

---

## 🎯 核心功能

| 模块 | 功能说明 |
|------|----------|
| **数据采集与预处理** | 读取 20 Newsgroups 数据集（18,456条原始数据），进行文本清洗、分词、去停用词、情感标注 |
| **朴素贝叶斯分类** | TF-IDF 特征提取 + 多项式朴素贝叶斯，准确率 **80.70%** |
| **TextCNN 深度学习** | 多卷积核捕捉局部语义特征，准确率 **90.64%** |
| **BERT 微调** | 加载本地预训练模型进行微调，准确率 **88.89%** |
| **多模型融合投票** | 三模型独立预测后投票集成，含兜底策略，准确率 **90.64%** |
| **聚类与观点挖掘** | K-Means 聚类（5个簇）+ PCA 降维可视化，识别不同观点群体 |
| **风险预警机制** | 基于负面比例和传播速度，自动判定绿/黄/红三级预警 |
| **可视化看板** | ECharts 绘制情感趋势图、词云图、PCA 散点图 |
| **智能研判** | 输入任意文本，实时返回三模型融合投票结果 |

---

## 🛠️ 技术栈

### 后端
| 技术 | 用途 |
|------|------|
| Flask 2.3 | Web 框架，提供 RESTful API |
| scikit-learn 1.3 | TF-IDF 特征提取、朴素贝叶斯、K-Means 聚类、PCA 降维 |
| PyTorch 2.0 | TextCNN 和 BERT 模型训练与推理 |
| Transformers 4.35 | 加载本地 BERT 预训练模型 |
| jieba 0.42 | 中文分词 |
| pandas / numpy | 数据处理 |

### 前端
| 技术 | 用途 |
|------|------|
| HTML + CSS | 页面结构与样式 |
| ECharts 5.4 | 图表渲染（趋势图、词云图、散点图） |
| jQuery 3.6 | Ajax 异步数据请求 |

### 数据与模型
- **数据集**：20 Newsgroups（20个新闻组类别）
- **文本处理**：分词 → 去停用词 → TF-IDF 向量化
- **深度学习模型**：TextCNN（卷积神经网络）
- **预训练模型**：`paraphrase-multilingual-MiniLM-L12-v2`（多语言 BERT）

---

## 📁 项目结构

```
public-opinion-monitoring-system/
│
├── 📁 20news-bydate/                     # 原始数据集
│   ├── 20news-bydate-train/              # 训练集（~11,000条）
│   └── 20news-bydate-test/               # 测试集（~7,000条）
│
├── 📁 clustering/                        # 聚类分析模块
│   └── cluster_analysis.py               # K-Means + PCA 聚类分析
│
├── 📁 data/                              # 数据预处理模块
│   ├── process_data.py                   # 数据读取、清洗、分词、标注
│   ├── balance_data.py                   # 数据平衡（上采样+下采样）
│   ├── processed_data.csv                # 预处理后数据（800条）
│   └── balanced_data.csv                 # 平衡后数据（852条，三类各284条）
│
├── 📁 models/                            # 模型定义模块
│   ├── naive_bayes_model.py              # 朴素贝叶斯分类器
│   ├── textcnn_model.py                  # TextCNN 深度学习模型
│   ├── bert_model.py                     # BERT 加载与预测
│   ├── ensemble_model.py                 # 多模型融合投票
│   └── 📁 saved_models/                  # 训练好的模型权重
│       ├── naive_bayes_balanced.pkl      # 朴素贝叶斯 ✅
│       ├── textcnn_balanced.pth          # TextCNN ✅
│       └── bert_finetuned.pth            # BERT ✅
│
├── 📁 paraphrase-multilingual-MiniLM-L12-v2/  # 本地 BERT 预训练模型
│
├── 📁 risk/                              # 风险预警模块
│   └── risk_alert.py                     # 绿/黄/红三级预警
│
├── 📁 static/                            # 静态资源
│   └── css/style.css                     # 自定义样式
│
├── 📁 templates/                         # 前端 HTML 模板
│   ├── index.html                        # 总览看板页
│   ├── cluster.html                      # 聚类分析页
│   └── predict.html                      # 智能研判页
│
├── app.py                                # Flask 主程序 ⭐
├── config.py                             # 系统配置文件
├── train_models_with_bert_full.py        # 完整训练脚本 ⭐
├── requirements.txt                      # Python 依赖清单
└── README.md                             # 项目说明文档

## 🚀 快速启动

### 1. 克隆项目

```bash
git clone https://github.com/tianj6807-ops/public-opinion-monitoring-system.git
cd public-opinion-monitoring-system
```

### 2. 创建虚拟环境（可选）

```bash
conda create -n opinion python=3.10
conda activate opinion
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 数据预处理

```bash
python data/process_data.py
```

### 5. 数据平衡（可选）

```bash
python data/balance_data.py
```

### 6. 训练模型

```bash
python train_models_with_bert_full.py
```

> ⏱️ TextCNN 训练约 5-10 分钟，BERT 微调需 10-30 分钟（取决于硬件）

### 7. 启动 Web 系统

```bash
python app.py
```

### 8. 访问系统

浏览器打开：**http://localhost:5000**

---

## 📊 模型性能

### 各模型准确率对比

| 模型 | 准确率 | 说明 |
|------|--------|------|
| 朴素贝叶斯 | **80.70%** | 基线模型，速度快、可解释性强 |
| TextCNN | **90.64%** | 卷积神经网络捕捉局部特征 |
| BERT（微调） | **88.89%** | 预训练 Transformer 理解上下文 |
| **三模型融合** | **90.64%** | 投票集成，兼顾准确性与鲁棒性 |

### 分类报告（融合模型）

| 类别 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 正面 | 0.76 | 0.96 | 0.85 |
| 中性 | 0.98 | 0.88 | 0.93 |
| 负面 | 0.98 | 0.82 | 0.90 |
| **加权平均** | **0.91** | **0.89** | **0.89** |

### 混淆矩阵

```
            预测正面   预测中性   预测负面
实际正面       50         7         0
实际中性        1        55         1
实际负面        0        10        47
```

---

## 📈 数据说明

| 项目 | 说明 |
|------|------|
| **数据集** | 20 Newsgroups（20 个新闻组类别） |
| **原始数据量** | 18,456 条 |
| **采样后数据量** | 800 条 |
| **平衡后数据量** | 852 条（正面 284 + 中性 284 + 负面 284） |
| **情感标注方式** | 基于关键词规则匹配 |

---

## ⚠️ 大文件说明

由于 GitHub 单个文件大小限制（≤100MB），以下大文件**未包含在仓库中**：

| 文件/文件夹 | 大小 | 说明 |
|------------|------|------|
| `paraphrase-multilingual-MiniLM-L12-v2/` | ~448MB | BERT 预训练模型 |
| `models/saved_models/bert_finetuned.pth` | ~448MB | BERT 微调权重 |
| `models/saved_models/textcnn_balanced.pth` | ~XXMB | TextCNN 训练权重 |
| `20news-bydate/` | ~XXMB | 20 Newsgroups 原始数据集 |

**本地运行正常**：这些文件在本地电脑 `D:\网络舆情\qimo\` 目录中完整存在。

**其他电脑运行**：如需在新环境中运行，请联系作者获取大文件。

---

## 📸 系统截图

### 总览看板
<img width="1842" height="816" alt="a67eeb3486bfca7547daaaa3fdfd2e83" src="https://github.com/user-attachments/assets/9257106e-2a82-4011-a2b8-60b630d9d2ca" />
<img width="2359" height="805" alt="a79ea08357fdf049e272dcd31fa75d63" src="https://github.com/user-attachments/assets/055f236c-7d34-4490-8204-7053b0dabe49" />
<img width="2316" height="927" alt="059663494c72b507e44d8e2581cf99d1" src="https://github.com/user-attachments/assets/6294cfed-5cfe-444b-9120-8715b437b046" />

### 聚类分析
<img width="2427" height="1200" alt="8714e90b761357f0cc99d14edef909eb" src="https://github.com/user-attachments/assets/8ccd7fb9-8077-4f18-a03a-52c2eb34e7dc" />

### 智能研判
<img width="2352" height="953" alt="5954fa8621d8184d6e8bc3b14b2e1c36" src="https://github.com/user-attachments/assets/9a2a763b-3322-490e-975d-4d0ead4f3f77" />
<img width="2304" height="976" alt="a4b36a056b631d9b5930f18e363d8172" src="https://github.com/user-attachments/assets/3b618f02-2978-4afe-90f5-0e83e20c3d56" />
<img width="2462" height="996" alt="c783f6dbf6886a5c16d4bc99dde2f5ca" src="https://github.com/user-attachments/assets/4f2c8be1-3399-4115-8095-af99a62a8c87" />
<img width="2379" height="800" alt="8b2b22838ad9731ce9a7e3b7be69ecf0" src="https://github.com/user-attachments/assets/4a866fc0-6196-493d-bc50-5c6de05298bb" />
<img width="1970" height="1143" alt="4962925fc6f5c1504a1514e17639cee2" src="https://github.com/user-attachments/assets/17a1fef3-0b09-46c9-b3fe-ae7a1f4beee2" />
<img width="1866" height="925" alt="a585c6e57c3bfa4f31baf5fbae6a4028" src="https://github.com/user-attachments/assets/fb2a1279-8916-4a94-99ce-20c6d8dd9487" />

---

## 📄 License

本项目采用 MIT License，可自由使用、修改和分发。

---


## 📚 参考资料

1. 20 Newsgroups Dataset：http://qwone.com/~jason/20Newsgroups/
2. BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding
3. TextCNN：Convolutional Neural Networks for Sentence Classification
4. Flask 官方文档：https://flask.palletsprojects.com/
5. ECharts 官方文档：https://echarts.apache.org/

---

**⭐ 如果这个项目对您有帮助，欢迎 Star！**
```
