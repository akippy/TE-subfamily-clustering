# Example data

このディレクトリには、公開用IDへ置換した2つの実データ例を収録しています。

## family 372

- 用途：Pythonだけで実行できるQuick Start
- 配列数：122
- 既知subfamily：LTR30、LTR30N1

## family 308

- 用途：SWIPEと`makeblastdb`を使用するFull Recursive Example
- 配列数：233
- 既知subfamily：LTR13A、LTR13_v
- 主実装で再帰的な追加分割が採用される例

各familyには次のファイルがあります。

- `pairwise_ml_distance.mldist`：IQ-TREE2 pairwise ML distance
- `alignment.fasta`：consensus・再割当検証用MSA
- `labels.csv`：`sequence_id,known_subfamily`のみ
- `metadata.json`：由来と加工内容

元のDfam accession・ゲノム座標を含むIDは収録していません。詳細は [../../docs/data_sources.md](../../docs/data_sources.md) を参照してください。
