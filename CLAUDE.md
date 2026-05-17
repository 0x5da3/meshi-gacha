# meshi-gacha

予算内でランダムに注文が決まるガチャ風 Web アプリ。実体は単一ファイル `index.html`
（HTML/CSS/JS 一体）。ビルド工程はなく、ブラウザで開けば動作する。

## 開発メモ

- 店舗データは `index.html` 内の `STORES` オブジェクトで定義。各店舗に
  `name / short / emoji / menu / theme` を持つ。
- `theme`（`rosso / rossoDark / verde / oro`）は `applyStore()` で CSS 変数に反映され、
  筐体（ガチャマシン）の配色が店舗ごとに切り替わる。

## 運用方針

- このリポジトリでは、ユーザーへの都度確認なしにブランチ／PR を `main` へ
  マージしてよい（事前承認済み）。それ以外のリスクのある操作（force push、
  履歴の破壊的変更、`main` への直接 push など）は従来どおり都度確認する。
