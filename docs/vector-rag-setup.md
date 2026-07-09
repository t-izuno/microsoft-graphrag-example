# Vector RAGセットアップ手順

本書は、GraphRAGとの比較用に本リポジトリへ追加した、従来型の
Vector RAG（チャンク分割＋埋め込み検索＋生成）パイプラインの
セットアップと実行方法をまとめたものです。GraphRAGと同じ入力
（リポジトリ直下の`input/`）に対して動作します。

GraphRAG本体のセットアップは先に
[uvを使ったGraphRAGセットアップ手順](./setup.md)を参照してください。

## 前提条件

`litellm`・`lancedb`・`tiktoken`・`typer`・`pyyaml`・`pydantic`・
`python-dotenv`は`pyproject.toml`の直接依存として追加済みです。
`uv sync`（または`uv add`実行時）で自動的にインストールされます。

## 設定ファイル（vector_rag/settings.yaml）

GraphRAGの`graphrag/settings.yaml`とは別に、`vector_rag/settings.yaml`
で完結した設定を持ちます。

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
  db_uri: output/lancedb
  table_name: chunks

retrieval:
  top_k: 10
```

- `input.base_dir`と`vector_store.db_uri`は、この設定ファイル自身の
  場所（`vector_rag/`）からの相対パスとして解決されます。既定値の
  `../input`はリポジトリ直下の共有`input/`を指します。
- `${GRAPHRAG_API_KEY}`は`graphrag/.env`から読み込まれ、GraphRAG側と
  同じAPIキーを再利用します。
- **GraphRAGとVector RAGを公平に比較するため、
  `completion_model`・`embedding_model`の`model_provider`と`model`は
  `graphrag/settings.yaml`と必ず同じ値にしてください。**
  Ollamaなどローカルモデルへ切り替える場合は、
  [Ollama設定手順](./ollama-setup.md)を参考に両方の設定ファイルを
  同じ内容に変更してください。

## チャンク分割について

GraphRAG本体の`TokenTextSplitter`
（`graphrag/index/text_splitting/text_splitting.py`）と全く同じ
アルゴリズムで分割します。文書ごとに全文を1回だけ`tiktoken`で
トークンID化し、`chunking.size`トークンずつ切り出し、
`chunking.size - chunking.overlap`トークンずつ開始位置をずらしながら
末尾まで繰り返す、固定長スライディングウィンドウ方式です。文単位・
段落単位などの意味的な区切りは考慮しません。両パイプラインで
チャンクの粒度をそろえることで、比較結果への影響を避けています。

## インデックスを作成する

`input/`ディレクトリ配下の`.txt`ファイルを解析し、チャンク分割・
埋め込みを行って`vector_rag/output/lancedb`に格納します。

`python -m`はカレントディレクトリを`sys.path`に追加する仕様のため、
**リポジトリ直下（`vector_rag/`ディレクトリの外）**で実行してくだ
さい。`vector_rag/`ディレクトリの中で実行すると
`No module named vector_rag`になります（実機で確認済みです）。

```bash
uv run python -m vector_rag index
```

## クエリを実行する

インデックス済みのチャンクから類似度上位（既定`top_k: 10`）を取得し、
それを根拠に回答を生成します。こちらもリポジトリ直下から実行して
ください。

```bash
uv run python -m vector_rag query "このリポジトリのテーマは何ですか"
```

## トラブルシューティング

- `No module named vector_rag`エラーが出る場合は、`vector_rag/`
  ディレクトリの中など、リポジトリ直下以外の場所で実行していないか
  確認してください。
- 埋め込み・生成でエラーが出る場合は、`vector_rag/settings.yaml`の
  `model_provider`・`model`・`api_key`が正しいか確認してください。
- インデックスが空のまま検索結果が0件になる場合は、先に
  `uv run python -m vector_rag index`を実行済みか確認してください。

## 参考資料

- [uvを使ったGraphRAGセットアップ手順](./setup.md)
- [OllamaをLLMとして使う設定手順](./ollama-setup.md)
- `graphrag/index/text_splitting/text_splitting.py`（インストール済み
  graphragパッケージ内、チャンク分割アルゴリズムの参照元）
