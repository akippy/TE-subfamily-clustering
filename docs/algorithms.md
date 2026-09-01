# アルゴリズム

## 距離行列

入力はIQ-TREE2のpairwise maximum-likelihood distance（`.mldist`）です。読み込み時に次を検証します。

- 1行目の配列数と実際の行数
- 各行の距離値数
- sequence IDの一意性
- 行列の正方性・対称性
- 欠損値・無限値・負値がないこと

## Affinity Matrix

family内最大距離を用いて次の変換を行います。

\[
W_{ij}=1-D_{ij}/D_{\max}
\]

その後、元研究実装と同じく対角成分を0へ設定します。familyをまたいだglobal scalingは行いません。

## 通常のSpectral Clustering

1. precomputed Affinity Matrixを読み込む。
2. 元研究コードと同じ成分数でSpectral Embeddingを計算する。
3. 各候補 `k` について先頭 `k` 成分を使用する。
4. K-meansでmembershipを計算する。
5. 各配列のSilhouette Scoreとその平均を計算する。
6. 平均Silhouette Scoreが最大の `k` を採用する。
7. 既知ラベルが与えられた場合はARIを計算する。

通常版ではSpectral処理の乱数シードを1、K-meansの乱数シードを0に固定します。これは元研究コードで使用した条件を維持するためです。

## SilhouetteベースのRecursive Spectral Clustering

各ノードで次の処理を行います。

1. ノード内配列数を `n` とする。
2. `k=2` から `min(20, floor(n / 10))` までを候補とする。
3. 各候補についてSpectral EmbeddingとK-meansを実行する。
4. 平均Silhouette Scoreの高い順に候補を並べる。
5. 上位3候補を順番に配列ベースの条件で検証する。
6. 最初に全条件を満たした候補だけを採用する。
7. 採用されたchild clusterへ同じ処理を再帰的に適用する。

### 分割採用条件

- すべてのchild clusterが10配列以上である。
- child consensus間のsimilarityがすべて0.95未満である。
- child consensus間similarityが親レイヤーから渡された下限以上である。
- SWIPEによるconsensusへの再割当精度が0.95以上である。
- 深さが5未満である。

### Consensusの作成

ノード内のaligned sequenceについて、各MSAカラムの最頻文字をconsensusとします。consensusがgapとなったカラムは、consensusと各copyの双方から除去します。

### Consensus similarity

2つのconsensusを次のスコアでglobal alignmentします。

- match：5
- mismatch：-4
- gap open：-10
- gap extension：-0.5

identityの計算では、どちらか一方がgapであるカラムを分母・分子の両方から除外します。

### SWIPE再割当

各child consensusからBLAST database version 4を作成し、copyをSWIPEで検索します。各copyの最高スコアhitがSpectral Clusteringのclusterと一致する割合を再割当精度とします。

### 階層ラベル

採用された分割をpathとして記録します。

```text
0
1-0
1-1
```

分割が一度も採用されなかった配列は `0` とします。

## 固定二分割ベースライン

固定二分割版では、各ノードで候補探索を行わず `k=2` のみを評価します。配列数、consensus similarity、レイヤー整合性、SWIPE再割当精度は主実装と同じ目的の条件を使用します。

主実装と比較する際に、分割条件の違いだけでなく、K-means乱数シードと元コードの再帰カウンタ設定も維持しています。
