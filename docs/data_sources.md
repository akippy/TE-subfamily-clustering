# サンプルデータの出典

## 概要

公開サンプルはHomo sapiensのTE family 372および308です。

| 項目 | 内容 |
|---|---|
| 生物種 | Homo sapiens |
| NCBI taxonomy ID | 9606 |
| 参照ゲノム | GRCh38.p14 |
| Assembly accession | GCA_000001405.29 |
| 既知subfamily | Dfam 3.9 curated families |
| MSA | MAFFT由来 |
| pairwise distance | IQ-TREE2 `.mldist` |

family 372は122配列で、既知ラベルはLTR30が83、LTR30N1が39です。family 308は233配列で、LTR13Aが173、LTR13_vが60です。

## 公開時の加工

元のsequence IDにはDfam accession、参照配列accession、GRCh38.p14上のゲノム座標が含まれていました。公開サンプルでは、IDだけを次の形式へ置換しました。

```text
family372_seq001
family308_seq001
```

同じ対応を以下へ適用しています。

- `.mldist`の行ID・列順序
- alignment FASTA
- `labels.csv`
- cluster membership
- recursive clustering出力
- expected output

元IDとの対応表は公開していません。変換スクリプトと検証スクリプトは `scripts/` に収録しています。

変換前後について、以下が一致することを検証しました。

- `.mldist`のすべての距離文字列
- aligned FASTAの全塩基・gap
- レコード順序
- 既知subfamilyラベル
- Quick StartとFull Recursiveのクラスタリング結果

## 配布していないもの

- GRCh38.p14全ゲノム
- RepeatMasker出力
- Onecodetofindtheallの中間生成物
- 元のannotation全列
- 元IDと公開IDの対応表
- family 372・308以外の配列・距離行列
- 910 familyのfamily別クラスタリング出力

## 利用条件

DfamはデータとソフトウェアをCC0で提供していますが、第三者由来のライブラリには個別条件があり得ます。

- [Dfam: About](https://dfam.org/about)

NCBIはデータ利用・配布に原則として制限を設けていないと説明していますが、投稿者が権利を主張する可能性も示しています。

- [NCBI policies and disclaimers](https://www.ncbi.nlm.nih.gov/home/about/policies/)
- [NCBI: About GenBank](https://www.ncbi.nlm.nih.gov/genbank/about/)
- [NCBI Genome Reference Consortium: Human](https://www.ncbi.nlm.nih.gov/grc/human)

このリポジトリ自体にはLICENSEを付与していません。外部データ・ソフトウェアには各提供元の条件が適用されます。
