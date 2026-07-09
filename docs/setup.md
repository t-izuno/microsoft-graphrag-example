# uvを使ったGraphRAGセットアップ手順

本書は、本リポジトリをクローンした状態からGraphRAGを使えるように
するための手順です。`graphrag/`ディレクトリに設定ファイル・プロンプト
一式はすでに含まれているため、[GraphRAG公式ドキュメントのGetting
Started](https://microsoft.github.io/graphrag/get_started/)にある
`graphrag init`からのセットアップは不要です。

## 前提条件

- `uv`がインストール済みであること
- Pythonバージョンについて

  公式Getting Startedでは「Python 3.10〜3.12が必須」と案内されていますが、
  これは記述時点のバージョンに基づくものです。PyPI上の
  `graphrag`パッケージ（3.1.0）のメタデータでは`requires-python`が
  `>=3.11,<3.14`と定義されています。本リポジトリの`.python-version`が
  示す3.13は、この範囲内にあるためそのまま利用できます。

動作確認済みのバージョン（本書作成時点）は以下のとおりです。

| ツール | バージョン |
| --- | --- |
| uv | 0.11.27 |
| Python | 3.13 |
| graphrag | 3.1.0 |

## 1. 依存関係を同期する

```bash
uv sync
```

`uv sync`は`pyproject.toml`と`uv.lock`の内容に基づいて、
リポジトリ直下の`.venv`に依存関係一式を解決・インストールします。
以後のコマンドは`uv run <コマンド>`で実行することで、
`.venv`を手動で有効化（`source .venv/bin/activate`）しなくても
同じ依存関係環境で実行できます。

## 2. .envにAPIキーを設定する

`graphrag/.env`は秘密情報を含むため`.gitignore`対象になっており、
クローンしただけのリポジトリには含まれていません。GraphRAGの
設定読み込み処理（`graphrag_common.config.load_config`）は
`.env`を必ず`settings.yaml`と同じディレクトリから探す仕様のため
（実機のソースコードで確認済み）、リポジトリ直下ではなく
`graphrag/.env`に置く必要があります。

`graphrag/.env.example`をコピーして`graphrag/.env`を作成し、
実際に使うLLMプロバイダーのAPIキーに書き換えてください。

```bash
cp graphrag/.env.example graphrag/.env
```

Ollamaなどキー認証が不要なローカルモデルを使う場合の扱いは
[Ollama設定手順](./ollama-setup.md)を参照してください。

## 3. 入力データを配置する

リポジトリ直下の`input/`ディレクトリ（GraphRAGとVector RAGで共有）に
解析対象のテキストファイルを配置します。すでに`input/book.txt`が
含まれている場合は、そのまま動作確認に使えます。

## 4. インデックスを作成する

`input/`ディレクトリ配下のテキストを解析し、グラフ構造や
コミュニティレポートを生成します。

```bash
uv run graphrag index --root graphrag
```

処理が完了すると、`graphrag/output/`ディレクトリにParquet形式の
成果物が生成されます。

## 5. クエリを実行する

生成したインデックスに対して質問を投げかけます。検索方式は
`--method`オプションで切り替えます。

```bash
# グローバル検索（コミュニティ全体を俯瞰した回答）
uv run graphrag query --root graphrag --method global \
  "このリポジトリのテーマは何ですか"

# ローカル検索（特定エンティティに近い詳細な回答）
uv run graphrag query --root graphrag --method local \
  "このリポジトリのテーマは何ですか"
```

## トラブルシューティング

- `uv run graphrag ...`が見つからない場合は、`uv sync`が
  完了しているか確認してください。
- 依存関係を追加・変更したい場合は`uv add <パッケージ名>`を使い、
  手動で`pyproject.toml`を編集しないようにしてください。
- コマンドで`--root graphrag`を付け忘れると、カレントディレクトリに
  `settings.yaml`が見つからずエラーになります。
- `GRAPHRAG_API_KEY`が未設定のままだと、インデックス作成・クエリ
  実行時に認証エラーになります。`graphrag/.env`の内容を確認して
  ください。

## 参考資料

- [Getting Started - GraphRAG](https://microsoft.github.io/graphrag/get_started/)
- [Detailed Configuration - GraphRAG](https://microsoft.github.io/graphrag/config/yaml/)
- [Working on projects - uv](https://docs.astral.sh/uv/guides/projects/)
