# TE-subfamily-clustering

トランスポゾン（Transposable Element; TE）のサブファミリーを、pairwise maximum-likelihood distanceとSpectral Clusteringを用いて教師なし分類する研究コードです。

大学院研究で実際に使用している解析処理のうち、距離行列の読み込み、Affinity Matrixへの変換、通常のSpectral Clustering、Recursive Spectral Clustering、既知サブファミリーとの評価を、第三者が追跡しやすい形に再構成しています。論文の完全再現用リポジトリではなく、研究で開発したクラスタリング処理の中核を示す公開版です。

## はじめに：何を分類する研究か

TE（Transposable Element）は、ゲノム内を移動・増幅してきた反復DNA配列群です。似た配列の大きなまとまりを **family**、その内部にある、より近縁な小グループを **subfamily** と呼びます。本研究では、同じfamilyに属する多数の配列を、既知subfamilyラベルを見ずに自動で分ける方法を検討しています。

- pairwise ML distanceは配列間の進化的な隔たりで、**小さいほど近い**ことを表します。
- Affinityはクラスタリングに用いる類似度で、**大きいほど似ている**ことを表します。
- Silhouette Scoreは、同じcluster内のまとまりとcluster間の分離を基に、クラスタ構造の妥当性を評価する指標です。
- Adjusted Rand Index（ARI）は予測clusterと既知subfamilyの一致度です。**1に近いほど一致**しており、既知ラベルはこの事後評価にだけ使用します。

## 入力・出力と2つの実行例

| 実行例 | 役割 | クラスタリングに使う入力 | 評価だけに使う入力 | 主な出力 |
|---|---|---|---|---|
| family 372 / Quick Start | Pythonだけで通常版を実行 | `.mldist` | `labels.csv` | 距離行列、Affinity Matrix、候補k、membership、ARI |
| family 308 / Full Recursive Example | 再帰分割と配列検証まで実行 | `.mldist`、alignment FASTA | `labels.csv` | Affinity Matrix、階層membership、分割診断、ARI |

`.mldist` は配列ペアごとの距離、alignment FASTAは同じfamily内の整列済み配列です。`labels.csv` にある既知subfamilyは、予測後にARIを計算するためだけに読み込みます。**教師なしクラスタリング、候補kの選択、再帰分割の採否には既知subfamilyラベルを使用しません。**

## 通常版とRecursive版の違い

| 観点 | 通常のSpectral Clustering | SilhouetteベースRecursive Spectral Clustering |
|---|---|---|
| 分割範囲 | family全体を一度だけクラスタリング | 採用されたchild clusterをさらに局所的に分割 |
| kの選び方 | family全体について候補kを探索 | 各ノードで候補kを探索し、Silhouette上位候補を順に検証 |
| 分割の検証 | 平均Silhouette Scoreでkを選択 | childサイズ、consensus similarity、親子レイヤー整合性、SWIPE再割当精度で採否を判断 |
| 結果 | 1階層のcluster番号 | 分割経路を保持した階層clusterラベル |
| 停止 | 1回のクラスタリングで終了 | 条件を満たす分割がない、または最大深さ5に達した時点で停止 |

通常版はfamily全体に適した1つの分割を求めます。Recursive版は、family全体の分割後も各childを独立に調べるため、局所的なsubfamily構造を表現できます。

## 実装範囲

この研究では、外部ツールと既存Pythonライブラリの数値計算を利用し、その入出力をつなぐ解析ロジックと再帰アルゴリズムの制御をPythonで実装しました。

### 外部ツールが担当する処理

| ツール | 研究全体での役割 | このリポジトリでの扱い |
|---|---|---|
| RepeatMasker | 上流のTE検出 | 実行コードは含めない |
| MAFFT | Multiple Sequence Alignment | 整列済みサンプルのみ収録 |
| IQ-TREE2 | 系統解析とpairwise ML distance算出 | `.mldist`サンプルのみ収録 |
| NCBI BLAST+ `makeblastdb` | child consensusの検索DB作成 | Full Recursive Exampleから呼び出す |
| SWIPE | copyをchild consensusへ再割当 | Full Recursive Exampleから呼び出す |

MSA前処理には、TEtrimmerの考え方を参考にした独自のカラムフィルタリングを使用しています。**TEtrimmer本体を実行しているわけではありません。** family構築と上流ツールの実行コード、大規模中間データは公開範囲に含めていません。

### Pythonライブラリが担当する処理

- scikit-learn：Spectral Embedding、K-means、Silhouette Score、ARIの数値計算
- NumPy / pandas：行列計算、表形式データの処理
- Biopython：FASTA入出力、consensus間のpairwise alignment

### Pythonで実装した処理

- IQ-TREE2 `.mldist`の読み込みと、ID・正方性・対称性・値域の検証
- family内最大距離を使ったAffinity Matrixへの変換
- 候補k探索の制御と、平均Silhouette Scoreに基づく候補選択
- Recursive Spectral Clusteringの再帰制御、分割採用・停止条件
- consensus作成・similarity計算を用いた分割検証の統合
- `makeblastdb` / SWIPEの呼び出し、出力解析、再割当精度の判定
- 分割経路を表す階層clusterラベルと診断ログの管理
- ARI評価、family単位の評価結果の集計、DBSCAN noiseの評価規則
- Quick Start / Full Recursive Exampleを実行するCLIとworkflow
- 元研究コードと照合して固定した期待値に対する回帰テスト

Spectral Embedding、K-means、Silhouette Score、ARIの数値アルゴリズム自体をフルスクラッチ実装したものではありません。これらはscikit-learnを使用し、本実装では候補探索、再帰制御、配列ベースの検証、入出力・評価を組み合わせています。

## 研究背景

同じfamilyの内部でも進化的に異なるsubfamilyが存在します。本研究では、配列間距離からその構造を推定し、通常版、再帰版、既存の比較手法を同一の既知ラベルに対して評価しています。

研究全体の処理フローは次のとおりです。

```text
TE配列
  ↓
TE familyの構築
  ↓
MAFFTによるMultiple Sequence Alignment
  ↓
MSAカラムのフィルタリング
  ↓
IQ-TREE2による系統解析・pairwise ML distance
  ↓
距離行列
  ↓
Affinity Matrix
  ↓
Spectral Clustering / Recursive Spectral Clustering
  ↓
既知Dfam subfamilyとの比較
  ↓
Adjusted Rand Index（ARI）等による評価
```

## 公開している処理

- IQ-TREE2 `.mldist` の読み込みと形式検証
- 距離行列からAffinity Matrixへの変換
- Silhouette Scoreでクラスタ数を選ぶ通常のSpectral Clustering
- Silhouette Scoreと配列検証を組み合わせたRecursive Spectral Clustering
- 固定 `k=2` で繰り返し分割する比較ベースライン
- 予測クラスタと既知Dfam subfamilyのARI評価
- family 372とfamily 308を使った小規模な実データ例
- 910 familyを対象とした7手法比較の再集計スクリプト

## Affinity Matrix

family内の最大距離を \(D_{\max}\) とし、距離行列 \(D\) を次式で反転正規化します。

\[
W_{ij} = 1 - \frac{D_{ij}}{D_{\max}}
\]

元研究コードの挙動を維持するため、変換後の対角成分は明示的に0へ設定します。

## 2種類のRecursive Spectral Clustering

主実装は、各ノードで候補 `k` を比較し、Silhouette Score上位の候補を配列ベースの条件で検証します。

| 項目 | 主実装 | 固定二分割ベースライン |
|---|---:|---:|
| 各ノードのクラスタ数 | Silhouette Scoreから選択 | 常に `k=2` |
| 検証する候補 | 上位3候補 | 1候補 |
| 最小childサイズ | 10 | 10 |
| consensus similarity | 95%未満 | 95%未満 |
| SWIPE再割当精度 | 95%以上 | 95%以上 |
| 再帰設定 | 最大深さ5 | 元実装の再帰カウンタ上限10 |
| K-means乱数シード | 1 | 0 |

主実装の詳細は [docs/algorithms.md](docs/algorithms.md)、固定二分割版は [fixed_binary_recursive.py](src/te_subfamily_clustering/baselines/fixed_binary_recursive.py) を参照してください。

## セットアップ

検証環境はPython 3.9.21です。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

通常のSpectral ClusteringまではPythonパッケージだけで実行できます。Full Recursive Exampleには、次のコマンドも必要です。

- `makeblastdb`：[NCBI BLAST+](https://www.ncbi.nlm.nih.gov/books/NBK569861/)に含まれます。
- `swipe`：[SWIPE（作者公開リポジトリ）](https://github.com/torognes/swipe)から入手できます。

導入後、`command -v makeblastdb` と `command -v swipe` の両方がパスを返すことを確認してください。詳細なインストール手順は各提供元を参照してください。

## Quick Start：family 372

family 372を使い、`.mldist`の読み込みからARI評価までをPythonだけで実行します。

```bash
python examples/quick_start.py
```

出力先は `outputs/quick_start_family_372/` です。

```text
pairwise_ml_distance.mldist
  ↓
distance_matrix.csv
  ↓
affinity_matrix.csv
  ↓
candidate_scores.csv
  ↓
cluster_membership.csv
  ↓
metrics.json
```

検証済みの結果は次のとおりです。

| 指標 | 値 |
|---|---:|
| 配列数 | 122 |
| 既知subfamily数 | 2 |
| Silhouette Scoreで選択されたk | 2 |
| 予測クラスタ数 | 2 |
| ARI | 0.966759 |

## Full Recursive Example：family 308

family 308では、consensus作成、SWIPE再割当検証、再帰的な再分割まで実行します。

```bash
command -v makeblastdb
command -v swipe
python examples/full_recursive.py
```

検証済みの結果は次のとおりです。

| 指標 | 値 |
|---|---:|
| 配列数 | 233 |
| 既知subfamily数 | 2 |
| 予測クラスタ数 | 3 |
| 採用された分割数 | 2 |
| ARI | 0.867126 |

固定 `k=2` ベースラインは次のコマンドで実行できます。

```bash
python examples/fixed_binary_baseline.py
```

公開用IDは `family372_seq001`、`family308_seq001` のような連番です。コードはsequence IDを一意な文字列として扱い、Dfam accessionやゲノム座標を含む元ID形式には依存しません。

## 910 familyでの評価

現行の研究出力について、7手法すべてで同じ910 familyが揃っていることを確認し、古い909 family・848 familyのsummaryを使用せず再集計しました。

- 全対象：910 family、746,565配列
- 既知subfamily数が2以上：73 family
- 上記のうちTE：63 family

既知subfamilyが複数あるTE 63 familyにおけるfamily別ARIの集計は次のとおりです。

| 手法 | Mean ARI | Median ARI |
|---|---:|---:|
| Sequence similarity（rule 80） | 0.0649 | 0.0031 |
| Phylogenetic tree（cutoff 0.02） | 0.0861 | 0.0551 |
| DBSCAN（eps 0.1） | 0.0814 | 0.0028 |
| UMAP + K-means | 0.3737 | 0.3384 |
| Spectral Clustering | 0.4717 | 0.4414 |
| Recursive Spectral（固定k=2） | 0.4051 | 0.4352 |
| Recursive Spectral（Silhouette） | 0.4113 | 0.4757 |

SilhouetteベースRecursive版のMedian ARIは0.4757で、通常のSpectral Clusteringの0.4414を上回りました。一方、Mean ARIは0.4113で通常版の0.4717を下回っており、再帰分割がすべてのfamilyで通常版を上回る結果ではありません。

この表を含む最小限の集約結果は [reports/evaluation_910_method_summary.csv](reports/evaluation_910_method_summary.csv) で確認できます。family別910件の結果は収録していません。評価定義、DBSCAN noiseの扱い、集計対象の詳細は [docs/evaluation.md](docs/evaluation.md) に記載しています。比較用のUMAP、DBSCAN、sequence similarity、phylogenetic tree分類の研究コードは、リポジトリの焦点を保つため公開していません。

## ディレクトリ構成

```text
src/te_subfamily_clustering/
├── io.py                    # .mldist、FASTA、labels、ID検証
├── affinity.py              # 距離からAffinity Matrixへの変換
├── spectral.py              # 通常のSpectral Clustering
├── recursive.py             # Silhouetteベースの再帰処理
├── sequence_validation.py   # consensus similarity、SWIPE再割当
├── evaluation.py            # ARIと集計
├── workflows.py             # 小規模なend-to-end workflow
└── baselines/
    └── fixed_binary_recursive.py

examples/
├── quick_start.py
├── full_recursive.py
├── fixed_binary_baseline.py
├── data/                    # ID変換済み実データ
└── expected/                # 公開用コードで再生成した期待出力

reports/
└── evaluation_910_method_summary.csv  # 910 family再集計の手法別集約値
```

## コードを読む順番

1. [io.py](src/te_subfamily_clustering/io.py)：入力形式とsequence IDの整合性検証
2. [affinity.py](src/te_subfamily_clustering/affinity.py)：距離からAffinity Matrixへの変換
3. [spectral.py](src/te_subfamily_clustering/spectral.py)：通常版の候補k探索と選択
4. [recursive.py](src/te_subfamily_clustering/recursive.py)：主実装の候補検証と再帰制御
5. [sequence_validation.py](src/te_subfamily_clustering/sequence_validation.py)：consensus similarityとSWIPE再割当
6. [evaluation.py](src/te_subfamily_clustering/evaluation.py)：ARIと集約指標
7. [fixed_binary_recursive.py](src/te_subfamily_clustering/baselines/fixed_binary_recursive.py)：固定 `k=2` の比較ベースライン

実行単位で全体を追う場合は [workflows.py](src/te_subfamily_clustering/workflows.py)、コマンド引数は [cli.py](src/te_subfamily_clustering/cli.py) を参照してください。アルゴリズムの詳細は [docs/algorithms.md](docs/algorithms.md) にまとめています。

## 入力形式

### `.mldist`

IQ-TREE2のpairwise ML distance形式です。1行目に配列数、以降はsequence IDと距離値が並びます。

```text
3
seq001 0.0 0.2 0.8
seq002 0.2 0.0 0.7
seq003 0.8 0.7 0.0
```

### `labels.csv`

```csv
sequence_id,known_subfamily
seq001,subfamily_A
seq002,subfamily_A
seq003,subfamily_B
```

`labels.csv` はARI評価専用であり、クラスタリング結果や候補kの決定には使いません。

Full Recursiveでは、同じsequence ID・同じ順序を持つaligned FASTAも必要です。不一致、重複、順序差は処理開始時にエラーにします。

## テスト

Pythonだけで完結するテスト：

```bash
python -m unittest discover -s tests -v
```

SWIPEと`makeblastdb`を含む統合テスト：

```bash
RUN_FULL_RECURSIVE_TESTS=1 python -m unittest discover -s tests -v
```

テストでは、ID整合性、距離変換、対角成分、通常版membership、主再帰版membership、固定二分割版membershipを確認します。

## サンプルデータ

サンプルはDfam 3.9の既知subfamily情報と、ヒト参照ゲノムGRCh38.p14から得たTE copyの派生データです。元のDfam accession・ゲノム座標を含むIDは、全ファイルで一貫した連番へ置換しています。距離値、aligned sequence、既知ラベル、レコード順序は変更していません。

詳細な出典と加工内容は [docs/data_sources.md](docs/data_sources.md) および [examples/data/README.md](examples/data/README.md) を参照してください。

## 制約

- 上流のfamily構築、RepeatMasker、MAFFT、IQ-TREE2処理はREADMEで説明する範囲です。
- 大規模な研究データや中間生成物は含めていません。
- サンプルはHomo sapiensの2 familyのみです。
- 本リポジトリは論文の完全再現環境ではありません。

## 利用について

現時点ではLICENSEファイルを付与していません。そのため、このリポジトリは第三者への自由な再利用・再配布を許諾するものではありません。外部データ・ツールには、それぞれの提供元の利用条件が適用されます。
