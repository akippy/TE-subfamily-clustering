# Expected outputs

ここにあるファイルは、公開用にID変換した入力を公開用コードへ与えて再生成した回帰テスト用出力です。古い研究結果CSVをそのままコピーしたものではありません。

検証時には、元研究コードの出力を元sequence IDで整列し、公開用IDを使った出力と比較しました。

- family 372：通常版membershipが全配列で一致し、候補kごとのARIは完全一致、Silhouette Scoreは浮動小数点許容誤差内で一致
- family 308：主再帰版membershipが全配列で一致し、44行の候補診断・採否が一致
- family 308：固定 `k=2` 版membershipが全配列で一致

これらの出力は `tests/` の回帰テストで使用します。
