# Microsoft GraphRAG Example

Microsoftが提唱する
[GraphRAG](https://microsoft.github.io/graphrag/)を使って高度な
RAG環境を構築し、その
「高度さ」が従来型のVector RAGと比べて実際にどれだけの違いを
生むのかを検証するリポジトリ。

## このリポジトリでやること

1. **GraphRAGで高度なRAG環境を構築する**（`graphrag/`）。ナレッジ
   グラフを構築し、local/global searchで質問に答える
2. **その「高度さ」を検証する**。同じ入力データに対して、比較対象と
   なるシンプルなVector RAG（チャンク分割＋埋め込み検索＋生成）を
   `vector_rag/`に用意し、同じ質問セットをGraphRAGとVector RAGの
   両方に投げてLLM-as-judgeで採点する仕組み（`evaluation/`）で
   差を確かめる

## リポジトリ構成

```text
graphrag/       GraphRAG本体の設定・プロンプト（settings.yaml, .env, prompts/）
vector_rag/     比較対象のVector RAGベースライン（settings.yaml, index/query CLI）
evaluation/     GraphRAGとVector RAGを比較検証するCLI（QA生成・実行・採点・レポート）
input/          両パイプラインで共有する入力データ
tests/          pytestによるユニットテスト
docs/           セットアップ・検証手順のドキュメント
```

## ドキュメント

まずGraphRAG環境を構築し、次に比較検証に必要なVector RAGと評価の
仕組みを用意する、という順番で進めます。

1. [uvを使ったGraphRAGセットアップ手順](docs/setup.md)
2. [Vector RAGセットアップ手順](docs/vector-rag-setup.md)
3. [GraphRAG対Vector RAG 評価手順](docs/evaluation.md)

LLMをOllamaに切り替える場合は
[OllamaをLLMとして使う設定手順](docs/ollama-setup.md)も参照してください。

## テスト

```bash
uv run pytest
```

## 参考資料

- [GraphRAG - 公式ドキュメント](https://microsoft.github.io/graphrag/)
- [GraphRAG (microsoft/graphrag) - GitHub](https://github.com/microsoft/graphrag)
