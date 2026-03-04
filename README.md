# 《從繼承到創新：協同演化雙通道架構下的強化學習策略融合》 (From Inheritance to Innovation: A Dual-Channel Genetic Algorithm for Deep Reinforcement Learning
BipedalWalker GA-Distillation and Dual-Channel Fusion experiments



---


## 📘 摘要
本研究探討一種結合 **基因演算法（Genetic Algorithm, GA）** 與 **策略蒸餾（Policy Distillation）** 的混合式強化學習架構，用以提升在 `BipedalWalker` 環境中的訓練穩定性與策略泛化能力。
研究中訓練兩個獨立的父母代理（Dad & Mom），並透過 **雙通道融合網路（Dual-Channel Fusion Network）** 生成子模型，使其能同時繼承並平衡雙方策略特徵。
整體系統整合了課程式難度調整、適應性交配與多教師蒸餾機制，以實現更穩定且高效的策略學習行為。


---


## 🧩 研究架構


```
父母模型 (PPO #1, PPO #2)
│
▼
[雙通道融合層]
│
▼
子模型 → 蒸餾訓練 → 微調 → 效能評估
```


**階段一：父母訓練** ─ 使用不同隨機種子與課程設定訓練兩個 PPO 專家模型，以獲得具差異化的行為策略。
**階段二：融合** ─ 透過雙通道融合機制，對應層級融合父母網路權重，生成初始子模型。
**階段三：蒸餾與微調** ─ 應用多教師策略蒸餾以穩定子模型學習，並透過少量互動資料進行細部修正。
**階段四：評估與可視化** ─ 分析父母與子模型之間的行為軌跡、分佈與報酬差異。


---


## 📊 實驗亮點


| 指標 | 父代 | 母代 | 子代 |
|:----:|:----:|:----:|:----:|
| 平均報酬 | 基準 (100%) | −8% | −5% |
| 動作穩定性 | ▲ 高波動 | ▲ 低波動 | ✅ 平衡 |
| 動作相似度 (MSE) | – | – | 約 0.84（父母平均） |
| 策略特徵 | 探索型 | 保守型 | 平衡型 |


📈 **關鍵觀察：**
- 子模型在時間序列曲線上呈現更平滑的控制，波動幅度明顯下降。
- 各動作維度的分佈集中且對稱，顯示行為一致性提升。
- 雙通道融合保留約 95% 父母平均報酬，同時大幅提升策略穩定度。


---


## ⚙️ 環境設定


```bash
git clone https://github.com/<your-username>/RL-NeuroEvolution-Project.git
cd RL-NeuroEvolution-Project
python3 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
電腦:dgx spark
```


執行訓練或分析：
```bash
python train.py
python analyze_action.py
```


---


## 🧠 技術特色


- **雙通道融合機制（Dual-Channel Fusion）**：以層級對應方式融合父母權重，保留語意結構並降低梯度衝突。
- **多教師蒸餾（Multi-Teacher Distillation）**：結合多個高效教師模型的策略知識，透過信心加權平均進行轉移。
- **課程式訓練（Curriculum Learning）**：根據環境難度動態調整訓練階段，防止早期崩潰。
- **動作分佈可視化（Action Distribution Visualization）**：比較各維度輸出分佈，以分析策略融合與一致性。
- **自動化實驗流程**：結合 Optuna 與 Ray 進行超參數搜尋與平行化訓練。


---


## 🧾 專案結構


```
RL-NeuroEvolution-Project/
│
├── train.py # 主要訓練流程（PPO / GA 混合）
├── fusion.py # 雙通道融合與權重對齊模組
├── distill.py # 多教師策略蒸餾模組
├── analyze_action.py # 動作相似度與分佈分析
├── requirements.txt # Python 相依套件
├── README.md # 專案說明文件
└── .gitignore # 忽略大型檔案（例如 .pth, log）
```


---


## 📚 關鍵詞
強化學習 · 神經演化 · 策略蒸餾 · 基因演算法 · 雙通道融合 · BipedalWalker · 課程式學習


---


## 👨‍💻 作者
**陳奕安（Alen Chen）**
國立中興大學 電機工程學系
Email: <your-email@example.com>
© 2025 Alen Chen — 版權所有。
