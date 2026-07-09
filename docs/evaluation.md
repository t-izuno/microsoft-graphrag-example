# GraphRAG対Vector RAG 評価手順

本書は、GraphRAG（local/global search）とVector RAGを同じ質問セットで
比較し、LLM-as-judgeで採点してレポートを作成する手順をまとめたもの
です。事前に[uvを使ったGraphRAGセットアップ手順](./setup.md)と
[Vector RAGセットアップ手順](./vector-rag-setup.md)を済ませ、
`uv run graphrag index --root graphrag`と
`uv run python -m vector_rag index`の両方を実行しておいてください。

以降の`uv run python -m evaluation ...`コマンドは、すべて
**リポジトリ直下**で実行してください。`python -m`はカレント
ディレクトリを`sys.path`に追加する仕様のため、`evaluation/`
ディレクトリの中など別の場所で実行すると
`No module named evaluation`になります（実機で確認済みです）。

## 1. 評価用QAセットの候補を生成する

`input/`配下のテキストをチャンク分割し、均等な間隔でサンプリングした
チャンクからLLMで質問・模範解答のペアを自動生成します。

```bash
uv run python -m evaluation generate-qa
```

`evaluation/qa_dataset.yaml`に、次の形式でエントリが書き出されます。

```yaml
- id: qa-001
  question: "質問文"
  expected_answer: "模範解答"
  source_chunk_id: "book-4"
  reviewed: false
```

## 2. 生成されたQAをレビューする

生成される質問・模範解答はLLMによる自動生成のため、そのままでは
品質にばらつきがあります。`evaluation/qa_dataset.yaml`を直接開き、
各エントリの`question`・`expected_answer`を確認・修正したうえで、
`reviewed`を`true`に変更してください。専用のレビューUIはなく、
ファイルの直接編集で運用します。`reviewed: true`にしたエントリのみ
が次の実行対象になります。

## 3. 実行・採点・レポート作成をまとめて行う

`evaluation/qa_dataset.yaml`内の`reviewed: true`エントリ**全件**を
対象に、次を1回のコマンドでまとめて実行します。1問ずつコマンドを
実行する必要はありません。

1. GraphRAG local search・GraphRAG global search・Vector RAGの3経路で
   回答を収集する
2. 期待回答と各手法の実際の回答をLLMに渡して1〜5点のスコアと採点
   理由を生成する（LLM-as-judge）
3. 手法ごとの平均スコアと質問別のスコア一覧をMarkdown表にまとめる

```bash
uv run python -m evaluation run [<run-id>]
```

`<run-id>`は実行を識別する文字列です（例: `2026-07-09-openai`）。
**省略すると現在時刻から自動生成されます**（例:
`20260709-153000`）。実行前に、`graphrag/settings.yaml`と
`vector_rag/settings.yaml`の`model_provider`・`model`が一致しているか
のチェックが走り、不一致があれば警告を表示します（警告のみで実行は
継続します）。`reviewed: true`のエントリが0件の場合は、その旨の
エラーメッセージを表示して終了します（手順2のレビューを済ませて
いるか確認してください）。

結果は次の3ファイルに保存されます。

- `evaluation/results/<run-id>.json`: 各手法の回答
- `evaluation/results/<run-id>-scored.json`: 採点結果
- `evaluation/results/<run-id>-report.md`: 比較レポート

### 特定の質問だけやり直す

一部の質問だけ回答・採点が失敗した場合や、レビュー後に質問文を
修正した場合など、全件を再実行せずに特定の質問だけやり直したい
ときは、同じ`<run-id>`に対して`--question-id`（`-q`）でIDを指定
します（複数指定可）。指定しなかった質問は、既存の
`evaluation/results/<run-id>.json`・`<run-id>-scored.json`の内容を
そのまま再利用します（LLM呼び出しをやり直さないため高速です）。

```bash
uv run python -m evaluation run <run-id> --question-id qa-005
```

## レポートの読み方

`evaluation/results/<run-id>-report.md`には次の2つの表が出力されます。

- **手法別 平均スコア**: GraphRAG（local/global）とVector RAGそれぞれの
  平均スコア（1〜5点）
- **質問別スコア**: 質問ごとに3手法のスコアを並べた一覧

平均スコアが高いほど、期待回答に近い回答を返せていることを示します。
ただしLLM-as-judgeの採点は完全に決定的ではなく、同じ入力でも実行の
たびにスコアが多少変動しうる点に注意してください。

## トラブルシューティング

- `model_provider`・`model`の不一致警告が出た場合、
  `graphrag/settings.yaml`と`vector_rag/settings.yaml`の
  `completion_model`・`embedding_model`を見直してください。比較の
  公平性のため、意図せず異なるモデルで比較しないよう注意してください。
- `evaluation run`でエラーが出る場合、GraphRAG・Vector RAG双方の
  インデックス作成（`graphrag index`・`vector_rag index`）が完了して
  いるか確認してください。
- `reviewed: trueのQAエントリがありません`というエラーで終了する
  場合は、手順2でレビューを済ませているか確認してください。

## 参考資料

- [uvを使ったGraphRAGセットアップ手順](./setup.md)
- [Vector RAGセットアップ手順](./vector-rag-setup.md)
- [OllamaをLLMとして使う設定手順](./ollama-setup.md)
