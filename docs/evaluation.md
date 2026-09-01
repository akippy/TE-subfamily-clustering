# 評価方法

## Adjusted Rand Index

予測クラスタと既知Dfam subfamilyの一致度をAdjusted Rand Index（ARI）で評価します。ARIはクラスタ番号そのものには依存しません。

既知ラベルはクラスタリングや `k` 選択には使用せず、評価時だけ使用します。ただし、研究上の参考指標であるSpectral upper boundは例外で、既知ラベルに対するARIが最大となる `k` を事後的に選びます。この値は主比較には使用しません。

## 910 familyの再集計

現行の以下7手法について、入力・出力family集合がすべて同じ910 familyであることを検証してから集計しています。

1. Sequence similarity（rule 80）
2. Phylogenetic tree（cutoff 0.02）
3. DBSCAN（`eps=0.1`, `min_samples=2`）
4. UMAP + K-means（Silhouette Scoreで `k` 選択）
5. Spectral Clustering（Silhouette Scoreで `k` 選択）
6. Recursive Spectral Clustering（固定 `k=2`）
7. Recursive Spectral Clustering（Silhouetteベース）

再集計スクリプトは `scripts/evaluate_dataset.py` です。`--expected-family-count 910` を必須指定し、いずれかの手法でfamilyが不足・過剰となる場合は集計を停止します。

公開している [手法別集約結果](../reports/evaluation_910_method_summary.csv) は、この条件で再集計した出力です。family別910件の詳細は公開対象に含めていません。

集計時に確認された対象は次のとおりです。

| subset | family数 | sequence数 |
|---|---:|---:|
| 全family | 910 | 746,565 |
| 既知subfamily数 > 1 | 73 | 132,937 |
| 既知subfamily数 > 1かつTE | 63 | 121,928 |

既知subfamilyが1種類しかないfamilyでは、ARIの解釈が複数subfamily familyと異なります。そのためREADMEの主な比較表には「既知subfamily数 > 1かつTE」の63 familyを使用しています。

## DBSCANのnoise

DBSCANの `-1` は通常のclusterではなくnoiseです。主比較のARIでは、各noise pointを互いに異なるsingleton clusterへ置き換えます。

補助指標として次の3種類を計算可能です。

- 全noiseを同じ `-1` clusterとして評価
- 各noiseを固有singleton clusterとして評価（主評価）
- noiseを除外して評価

## 集約指標

family別ARIから、手法ごとに以下を計算します。

- Mean ARI
- sequence数で重み付けしたMean ARI
- Median ARI
- 第1・第3四分位数
- ARI 0.9以上のfamily数
- ARI 0.5未満のfamily数

sequence-weighted Mean ARIはfamily別ARIの加重平均であり、全sequenceを一括して計算したARIではありません。
