# OllamaをLLMとして使う設定手順

GraphRAG（3.1.0以降）はモデル呼び出しに
[LiteLLM](https://docs.litellm.ai/)を使用しており、公式ドキュメントでも
「LiteLLMは100以上のモデルをサポートする」と案内されています。
本書では、ローカルで動作する[Ollama](https://ollama.com/)を
completion（チャット）モデルおよびembeddingモデルとして
利用するための設定方法をまとめます。

## 前提となる注意事項（公式ドキュメントに明記あり）

GraphRAG公式の[Language Model
Selection](https://microsoft.github.io/graphrag/config/models/)には、
Ollamaなどでモデル呼び出しをプロキシする場合について次の注意が
記載されています。

> Many users have used platforms such as ollama and LiteLLM Proxy
> Server to proxy the underlying model HTTP calls to a different
> model provider. However, there are frequently issues with malformed
> responses (especially JSON), so your model needs to reliably return
> the specific response formats that GraphRAG expects.

つまり、選択するモデルが構造化出力（JSON
スキーマ）を安定して返せることが前提条件です。小さすぎるモデルや
関数呼び出し・JSON出力に対応していないモデルでは、
インデックス作成中にJSON解析エラーが発生する可能性があります。
`llama3.1`や`qwen2.5`など、関数呼び出し・JSON出力に対応した
モデルを選ぶことを推奨します。

## 前提条件

- [Ollama](https://ollama.com/)をインストール済みで、
  `ollama serve`（デフォルトで`http://localhost:11434`）が
  起動していること
- 利用するモデルを事前に`ollama pull`しておくこと

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## 設定方法

GraphRAG側のモデル指定は`model_provider`と`model`の2つに
分かれており、公式ドキュメントには「`model_provider`が`/`より前の
部分、`model`が`/`より後ろの部分に対応する」（LiteLLMの
モデル文字列`<provider>/<model>`の分解）と説明されています。
この対応関係と、LiteLLM公式のOllamaプロバイダードキュメントを
突き合わせると、設定方法は大きく2通りが考えられます。

いずれの方法も、GraphRAG側でOllama利用の具体的な設定例が
公式ドキュメントに掲載されているわけではないため、
**未検証（要動作確認）の構成**として提示します。適用後は
`uv run graphrag index --root graphrag`を実行し、エラーが出ないか
確認してください。設定ファイルは`graphrag/settings.yaml`です。

### 方法A: LiteLLMのOllamaネイティブプロバイダーを使う

[LiteLLM公式のOllamaプロバイダードキュメント](https://docs.litellm.ai/docs/providers/ollama)
に基づく設定です。チャット用途では`/api/chat`エンドポイントに
対応する`ollama_chat`プレフィックスの使用が推奨されています。

```yaml
completion_models:
  default_completion_model:
    model_provider: ollama_chat
    model: llama3.1
    api_base: http://localhost:11434

embedding_models:
  default_embedding_model:
    model_provider: ollama
    model: nomic-embed-text
    api_base: http://localhost:11434
```

- `auth_method`・`api_key`はローカル利用のため省略できると
  考えられますが、GraphRAGの設定バリデーションでAPIキーが
  必須として扱われる場合は、`auth_method: api_key`と
  空でないダミー文字列（例:`ollama`）を`api_key`に設定してください。
- embeddingについては、LiteLLM側で`api_base`が正しく渡らない
  既知の不具合報告（[BerriAI/litellm#2674](https://github.com/BerriAI/litellm/issues/2674)、
  [BerriAI/litellm#7451](https://github.com/BerriAI/litellm/issues/7451)）が
  あるため、動作しない場合は方法Bを試してください。

### 方法B: OllamaのOpenAI互換エンドポイントを使う

Ollamaが提供する[OpenAI互換API](https://docs.litellm.ai/docs/providers/openai_compatible)
（`/v1`エンドポイント）を、`model_provider: openai`として
呼び出す方法です。LiteLLM公式ドキュメントでは「OpenAI互換
エンドポイントを使う場合もAPIキーは必須項目のため、
ダミー値を指定する」旨が案内されています。

```yaml
completion_models:
  default_completion_model:
    model_provider: openai
    model: llama3.1
    auth_method: api_key
    api_key: ollama # 実際には検証されないダミー値
    api_base: http://localhost:11434/v1

embedding_models:
  default_embedding_model:
    model_provider: openai
    model: nomic-embed-text
    auth_method: api_key
    api_key: ollama
    api_base: http://localhost:11434/v1
```

`api_base`には`/v1`サフィックスを含める点に注意してください
（LiteLLM公式ドキュメントで明示的に推奨されています）。

## .envファイルについて

上記どちらの方法でもOllama自体はAPIキー認証を行わないため、
`graphrag/.env`内の`GRAPHRAG_API_KEY`はダミー値のままで問題
ありません。`graphrag/settings.yaml`側で`${GRAPHRAG_API_KEY}`を
参照する代わりに、`api_key`欄に直接ダミー文字列を書いても動作します
（その場合は`.env`への依存がなくなります）。

## 動作確認

```bash
uv run graphrag index --root graphrag
```

インデックス作成中にJSON解析エラーやタイムアウトが発生する
場合は、次を見直してください。

- モデルが構造化出力・JSONスキーマに対応しているか（前提となる注意事項を参照）
- `ollama serve`が起動しており、`api_base`のポート番号が
  一致しているか
- `ollama pull`でモデルを取得済みか

## 参考資料

- [Language Model Selection - GraphRAG](https://microsoft.github.io/graphrag/config/models/)
- [Detailed Configuration - GraphRAG](https://microsoft.github.io/graphrag/config/yaml/)
- [Ollama - liteLLM](https://docs.litellm.ai/docs/providers/ollama)
- [OpenAI-Compatible Endpoints - liteLLM](https://docs.litellm.ai/docs/providers/openai_compatible)
- [Ollama公式サイト](https://ollama.com/)
