# GraphRAG対Vector RAG比較基盤 設計書

## 背景・目的

GraphRAGの評価として、同一の入力データに対して従来型のVector
RAG（チャンク分割＋埋め込み検索＋生成）と比較したい。そのために
以下を用意する。

- GraphRAGと同じ入力（`input/`）に対して動作するVector RAGパイプライン
- 入力から評価用QA（質問・期待回答）セットを作る仕組み
- GraphRAG（local/global search）とVector RAGの両方に同じ質問を
  投げ、回答を収集する実行基盤
- 期待回答と実際の回答をLLM-as-judgeで採点し、レポートにまとめる仕組み

## スコープ

含むもの:

- Vector RAGパイプライン本体（インデックス作成・クエリ）
- 評価用QAセットの自動生成（＋人間レビューの受け皿）
- GraphRAG（local/global）とVector RAGへの一括問い合わせ
- LLM-as-judgeによる採点とMarkdownレポート出力
- 上記に対応するドキュメント（セットアップ手順・評価手順）

含まないもの（YAGNI）:

- QAレビュー用の専用UI（ファイル直接編集で対応）
- 埋め込み類似度など、LLM-as-judge以外の採点方式
- CI上での自動LLM評価実行（手動実行のみを想定）

## ディレクトリ構成（確定）

既存の`settings.yaml`・`.env`・`prompts/`・`main.py`関連はGraphRAG
専用として`graphrag/`配下に移動し、`vector_rag/`と対称な構成にする。
`input/`はリポジトリ直下に残し、両パイプラインの設定ファイルから
それぞれ相対パスで参照する共有ディレクトリとする。コードは変更せず、
`graphrag/settings.yaml`の`input_storage.base_dir`と
`vector_rag/settings.yaml`の`input.base_dir`という設定値のみを
`../input`に向ける。

```text
graphrag/
  settings.yaml
  .env
  prompts/
    extract_graph.txt など（既存プロンプト一式）
  output/        (.gitignore対象、graphrag index の出力)
  cache/         (.gitignore対象)
  logs/          (.gitignore対象)

vector_rag/
  settings.yaml
  prompts/
    query_system_prompt.txt
  __init__.py
  config.py
  chunking.py
  embedding.py
  completion.py
  store.py
  indexer.py
  answer.py
  cli.py
  output/        (.gitignore対象、lancedbデータ)

input/            <- graphrag/とvector_rag/で共有

evaluation/
  __init__.py
  cli.py
  generate_qa.py
  graphrag_client.py
  vector_rag_client.py
  judge.py
  report.py
  config_check.py
  qa_dataset.yaml
  results/         (.gitignore対象、実行・採点結果のJSON)

docs/
  setup.md              (要修正: --root graphrag を明記)
  ollama-setup.md        (要修正: --root graphrag を明記)
  vector-rag-setup.md    (新規)
  evaluation.md          (新規)

main.py
pyproject.toml
```

当初の案では評価基盤のディレクトリ名を`eval/`としていたが、Python
組み込み関数`eval()`との混同を避けるため`evaluation/`に変更する。

## コンポーネント詳細

### vector_rag package

`graphrag`パッケージの推移的依存としてすでにインストール済みの
`litellm`（LLM呼び出し）・`lancedb`（ベクトルストア、GraphRAGの
`vector_store`設定と同じ技術）・`tiktoken`（チャンク分割）・
`typer`／`rich`（CLI）を、`pyproject.toml`の直接依存に昇格させて
利用する。新規の重い依存追加は発生しない。

- `config.py`: `vector_rag/settings.yaml`を読み込み、`pydantic`
  モデルへ変換する
- `chunking.py`: GraphRAGの`TokenTextSplitter`
  （`graphrag/index/text_splitting/text_splitting.py`）と同じ
  アルゴリズムで分割する。実際のソースコードで確認した挙動は次の
  とおり。
  1. 入力ファイル（ドキュメント）ごとに、テキスト全体を一度だけ
     `tiktoken`でトークンID列にエンコードする（複数ファイルを
     連結してからエンコードすることはしない）
  2. 先頭から`chunk_size`トークン分を1チャンクとして切り出す
  3. 開始位置を`chunk_size - chunk_overlap`トークン分だけ進め、
     末尾に達するまで2〜3を繰り返す
  4. 各チャンクのトークンID列をデコードしてテキストに戻す

     文単位・段落単位などの意味的な区切りは考慮しない、純粋な
     固定長スライディングウィンドウである。サイズ・オーバーラップ・
     エンコーディングは`vector_rag/settings.yaml`の`chunking`
     セクションで指定し、既定値はGraphRAGの`settings.yaml`と同じ値
     （size 1200 / overlap 100 / encoding_model o200k_base）にして
     チャンク粒度をそろえる
- `embedding.py`: `litellm.embedding()`のラッパー
- `completion.py`: `litellm.completion()`のラッパー
- `store.py`: `lancedb`への接続・テーブル作成・類似検索
- `indexer.py`: `input.base_dir`（共有`input/`）配下のテキストを
  読み込み、チャンク分割・埋め込みを行い`lancedb`に格納する
- `answer.py`: クエリを埋め込み、`store.py`でtop-k類似検索した
  チャンクと質問を`prompts/query_system_prompt.txt`のテンプレート
  に埋め込み、`completion.py`経由で回答を生成する
- `cli.py`: `typer`製CLI。`uv run python -m vector_rag index`と
  `uv run python -m vector_rag query "質問文"`を提供し、GraphRAGの
  `graphrag index`／`graphrag query`と対になる操作感にする

`vector_rag/settings.yaml`のスキーマ（案）:

```yaml
input:
  base_dir: ../input

completion_model:
  model_provider: openai
  model: gpt-4.1
  auth_method: api_key
  api_key: ${GRAPHRAG_API_KEY}

embedding_model:
  model_provider: openai
  model: text-embedding-3-large
  auth_method: api_key
  api_key: ${GRAPHRAG_API_KEY}

chunking:
  size: 1200
  overlap: 100
  encoding_model: o200k_base

vector_store:
  db_uri: vector_rag/output/lancedb
  table_name: chunks

retrieval:
  top_k: 10
```

パスの基準は`vector_rag/`パッケージのファイル自身
（`Path(__file__).parent`）を起点に解決し、GraphRAGのように
カレントディレクトリの変更（chdir）には依存しない。`input.base_dir`
も同じ基準（`Path(__file__).parent / "../input"`）で解決し、共有
`input/`を直接参照する。

### GraphRAGへの問い合わせ（Python API直接呼び出し）

`graphrag.api`（`local_search`／`global_search`）を直接呼び出す。
このAPIはGraphRAG自身のdocstringで「開発中であり後方互換性は
保証されない」と明記されているため、`evaluation/graphrag_client.py`
にバージョン依存箇所を局所化する。

`local_search`／`global_search`はentities・communities・
community_reports等のDataFrameを引数に取るが、これをparquet出力
から読み込むヘルパー（`graphrag.cli.query._resolve_output_files`）
はCLI内部のプライベート関数であり公開APIではない。そのため、
同等の処理を`graphrag_storage.create_storage`／
`create_table_provider`と`graphrag.data_model.data_reader.DataReader`
（いずれも公開クラス）を使って`evaluation/graphrag_client.py`内に
再実装する。GraphRAGのバージョンを上げた際はこのファイルの動作を
優先的に確認する、という注意書きをコード内コメントとdocsの両方に
残す。

`graphrag.config.load_config`はデフォルトで設定ファイルのある
ディレクトリへ`os.chdir`する（`set_cwd=True`）。この副作用は
プロセス全体に影響するため、`graphrag_client.py`では次の手順で
副作用を局所化する。

1. 呼び出し前のカレントディレクトリを保存する
2. `load_config(root_dir=<repo root>/"graphrag")`を呼ぶ
   （この時点でGraphRAGの設定内のパスは絶対パスへ解決される）
3. 保存しておいたカレントディレクトリへ直後に戻す
4. 以降の`DataReader`によるparquet読み込みや`local_search`／
   `global_search`の呼び出しは、絶対パス化済みの設定オブジェクトを
   使うため、カレントディレクトリの影響を受けない

`graphrag/settings.yaml`の`input_storage.base_dir`は`"input"`から
`"../input"`に変更する。コード自体は変更せず設定値のみの変更であり、
前述の`set_cwd=True`の挙動により`graphrag/`をrootにしたまま共有
`input/`を参照できる。

### 設定内容の整合性チェック

`evaluation/config_check.py`が、`graphrag/settings.yaml`の
`completion_models.default_completion_model`／
`embedding_models.default_embedding_model`と、
`vector_rag/settings.yaml`の`completion_model`／
`embedding_model`について、`model_provider`と`model`の値を比較する。
不一致があれば標準エラー出力に警告を表示するが、実行は継続する
（意図的にプロバイダーを変えて実験するケースを許容するため）。

### 評価用QAセットの生成（generate_qa.py）

`input/book.txt`を`vector_rag`と同じチャンク設定で分割し、
全チャンクからほぼ均等な間隔でサンプリングして30〜50件程度になる
ように件数を調整する（例: チャンク数120なら3個おきに40件抽出）。
各サンプルチャンクを`graphrag/settings.yaml`の`completion_model`で
LLMに渡し、「この一節の内容から質問と模範解答を1組作成して」という
指示でQAペアを1つ生成する。

出力は`evaluation/qa_dataset.yaml`に次の形式で書き出す。

```yaml
- id: qa-001
  question: "質問文"
  expected_answer: "模範解答"
  source_chunk_id: "chunk-004"
  reviewed: false
```

専用のレビューUIは作らず、人間が本ファイルを直接編集し
`reviewed: true`に変更したものだけを評価対象にする。

### 実行（run.py）とレポート（report.py）

`evaluation/qa_dataset.yaml`から`reviewed: true`の項目のみを対象に、
各質問について次の3種類の回答を収集する。

- GraphRAG local search（`evaluation/graphrag_client.py`経由）
- GraphRAG global search（同上）
- Vector RAG（`evaluation/vector_rag_client.py`が`vector_rag.answer`
  を直接呼び出す）

収集結果は`evaluation/results/<run-id>.json`に保存する
（質問ID・手法・回答本文・所要時間などを含む）。

`judge.py`は、保存した結果ファイルを読み込み、各回答について
期待回答と実際の回答をLLMに渡し、1〜5点のスコアと採点理由を
生成する。採点結果は`evaluation/results/<run-id>-scored.json`に
保存する。

`report.py`は採点結果を集計し、手法（GraphRAG local／GraphRAG
global／Vector RAG）ごとの平均スコアと質問別スコア一覧を
`evaluation/report.md`にMarkdown表として出力する。

`evaluation/cli.py`は`typer`製CLIとして
`uv run python -m evaluation generate-qa`／
`uv run python -m evaluation run`／
`uv run python -m evaluation report`の3サブコマンドを提供する。

## 依存関係の変更

`pyproject.toml`の`dependencies`に以下を明示的に追加する
（いずれも`graphrag`の推移的依存としてすでにインストール済みのため
新規ダウンロードは発生しない）。

- `litellm`
- `lancedb`
- `tiktoken`
- `typer`
- `pyyaml`
- `pydantic`

開発用依存として`pytest`を新規に追加する
（`uv add --dev pytest`）。

## ドキュメントの変更

- `docs/setup.md`: `graphrag/`ディレクトリへの移動に伴い、
  `uv run graphrag init --root graphrag ...`・
  `uv run graphrag index --root graphrag`・
  `uv run graphrag query --root graphrag --method local/global "..."`
  のように`--root graphrag`を明記するよう修正する
- `docs/ollama-setup.md`: 設定ファイルのパスを`graphrag/settings.yaml`
  に修正する
- `docs/vector-rag-setup.md`（新規）: Vector RAGパイプラインの
  セットアップ・`index`／`query`コマンドの使い方
- `docs/evaluation.md`（新規）: QAセット生成・人間レビュー・実行・
  採点・レポートの読み方

## テスト方針

現在リポジトリにテスト基盤がないため、`pytest`を新規導入する。

- `vector_rag`: チャンク分割・設定読み込み・lancedb格納検索を
  ユニットテストする（LLM呼び出しはモック化する）
- `vector_rag`: `index`／`query`のCLIについて、小さなダミー
  テキストを使ったスモークテストを行う
- `evaluation`: `graphrag_client.py`のDataFrame読み込み部分と
  `config_check.py`の整合性チェックロジックをユニットテストする
  （LLM呼び出し・GraphRAGインデックス実行はモック化する）
- 実LLMを使った統合的な動作確認は自動テストの対象外とし、手動実行
  で確認する

## 実装順序（フェーズ）

1. ディレクトリ再編成（`graphrag/`への移動、ドキュメント修正）
2. `vector_rag`パイプライン本体とドキュメント
3. `evaluation`（QA生成・実行・採点・レポート）とドキュメント

## 既知の制約・注意点

- `graphrag.api`は開発中のAPIであり、GraphRAGのバージョンアップ時に
  `evaluation/graphrag_client.py`の動作確認が必要になる可能性が
  高い
- LLM-as-judgeによる採点は完全に決定的ではないため、同じ入力でも
  実行のたびにスコアが多少変動しうる
