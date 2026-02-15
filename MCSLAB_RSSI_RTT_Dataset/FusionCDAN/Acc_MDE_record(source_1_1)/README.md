# Info
## 1. experiment_summary(old with data leakage)
code: fusion_cdan_v3.py
這個版本的 load_wifi_data() 存在輕微資料洩漏（數據不採用，但保留）

## 2. fusion_cdan_v3_summary
code: fusion_cdan_v3_fix_optimized.py
這個版本修復了以下要素：
1. load_wifi_data() 的資料洩漏
2. 選擇最佳模型的策略改成計算 source domain acc 以及 target domain classify entropy 的分數
3. Dataloader 以及其他小部分的程式優化

## 3. fusion_cdan_v4_summary
code: fusion_cdan_v4.py
這個版本加入了以下要素：
1. 在特徵萃取器與分類器之間，加入了 Gate Network 針對 RSSI、RTT 的特徵好壞，做動態的加權
2. 期望能達到更好的 fusion 效果