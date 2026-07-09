# TASKS

マイルストーン: M1
ゴール: GraphRAGとVector RAGを同一入力で比較できる基盤を構築する

## ワークフロールール

- タスク着手時にステータスを 🚧 に更新する
- タスク完了時にステータスを ✅ に更新する
- DependsOn のタスクがすべて ✅ でないタスクには着手しない

## ステータス表記ルール

| Status | 意味 |
| ---- | ----- |
| ⏳ | 未着手、TODO |
| 🚧 | 作業中、IN_PROGRESS |
| 🧪 | 確認待ち、REVIEW |
| ✅ | 完了、DONE |
| 🚫 | 中止、CANCELLED |

## タスク一覧

| ID | Status | Summary | DependsOn |
| ---- | ---- | ---- | ---- |
| TASK-001 | ✅ | 既存のgraphrag関連ファイルをgraphrag/配下へ移動する | - |
| TASK-002 | ✅ | graphrag設定のinput_storage.base_dirと.gitignoreを整備する | TASK-001 |
| TASK-003 | ✅ | docs/setup.mdをgraphrag/ルート前提の手順に修正する | TASK-002 |
| TASK-004 | ✅ | docs/ollama-setup.mdの設定ファイルパスを修正する | TASK-002 |
| TASK-005 | ✅ | pyproject.tomlに直接依存と開発依存pytestを追加する | - |
| TASK-006 | ✅ | vector_rag/settings.yamlのスキーマとconfig.pyを実装する | TASK-002,TASK-005 |
| TASK-007 | ✅ | chunking.pyを実装しユニットテストを書く | TASK-005 |
| TASK-008 | ✅ | embedding.py・completion.pyを実装する | TASK-006 |
| TASK-009 | ✅ | store.pyを実装しユニットテストを書く | TASK-006 |
| TASK-010 | ✅ | indexer.pyを実装する | TASK-007,TASK-008,TASK-009 |
| TASK-011 | ✅ | query_system_prompt.txtとanswer.pyを実装する | TASK-008,TASK-009 |
| TASK-012 | ✅ | cli.py（index/query）を実装しスモークテストを書く | TASK-010,TASK-011 |
| TASK-013 | ✅ | docs/vector-rag-setup.mdを作成する | TASK-012 |
| TASK-014 | ✅ | evaluation/config_check.pyを実装しユニットテストを書く | TASK-006 |
| TASK-015 | ✅ | evaluation/graphrag_client.pyを実装しユニットテストを書く | TASK-002 |
| TASK-016 | ✅ | evaluation/vector_rag_client.pyを実装する | TASK-011 |
| TASK-017 | ✅ | evaluation/generate_qa.pyを実装する | TASK-007 |
| TASK-018 | ✅ | evaluation/run.pyを実装する | TASK-014,TASK-015,TASK-016 |
| TASK-019 | ✅ | evaluation/judge.pyを実装する | TASK-018 |
| TASK-020 | ✅ | evaluation/report.pyを実装する | TASK-019 |
| TASK-021 | ✅ | evaluation/cli.py（3サブコマンド）を実装する | TASK-017,TASK-020 |
| TASK-022 | ✅ | docs/evaluation.mdを作成する | TASK-021 |

## タスク詳細（補足が必要な場合のみ）

### TASK-001

- 補足: 移動対象はsettings.yaml・.env・prompts/。input/とmain.pyは
  リポジトリ直下に残す
- 注意: prompts配下のファイル名・内容は変更しない

### TASK-002

- 補足: base_dirは"input"から"../input"へ変更する。
  graphrag/output・graphrag/cache・graphrag/logs・
  vector_rag/output・evaluation/results・.envを.gitignoreに追加する

### TASK-005

- 補足: litellm・lancedb・tiktoken・typer・pyyaml・pydantic・
  python-dotenv（.env読み込み用、実装中に追加で必要と判明）を追加した

### TASK-006

- 補足: vector_rag/settings.yamlのinput.base_dirは"../input"とする

### TASK-007

- 補足: GraphRAGのTokenTextSplitter
  （graphrag/index/text_splitting/text_splitting.py）と同じ
  アルゴリズムで実装する。文書ごとに全文を1回だけtiktokenで
  トークンID化し、chunk_sizeトークンずつ切り出し、
  chunk_size-chunk_overlapトークンずつ開始位置をずらしながら
  末尾まで繰り返す固定長スライディングウィンドウ方式。文単位・
  段落単位の区切りは考慮しない
- 補足: tiktokenのo200k_baseエンコーディングは初回ロード時に
  ネットワーク経由でダウンロードが必要で、この開発環境では取得
  できないことを確認した。GraphRAG自身のTokenTextSplitterと同様に
  encode/decode関数を注入可能にし、ユニットテストは実際のtiktokenを
  使わず簡易なフェイクトークナイザーで検証している

### TASK-015

- 補足: graphrag.cli.query._resolve_output_filesと同等の処理を
  graphrag_storage.create_storage/create_table_providerと
  graphrag.data_model.data_reader.DataReaderを使って再実装する。
  load_configのchdir副作用は呼び出し前後でカレントディレクトリを
  保存・復元して局所化する

### TASK-021

- 補足: tests/vector_rag/test_cli.pyとtests/evaluation/test_cli.pyが
  同名モジュールとして衝突したため、tests/配下に__init__.pyを追加して
  解消した

## Backlog一覧

| ID | Status | Summary | DependsOn |
| ---- | ---- | ---- | ---- |
| BACKLOG-001 | ⏳ | 埋め込み類似度による補助的な採点方式を追加する | - |
| BACKLOG-002 | ⏳ | vector_ragのチャンキング戦略を複数並列検証できるようにする | - |

## Backlog詳細（補足が必要な場合のみ）

### BACKLOG-001

- 補足: 現状はLLM-as-judgeのみを採点方式とする

### BACKLOG-002

- 補足: 現状はGraphRAGと同じ固定長スライディングウィンドウのみを
  採用する。将来的に文脈（意味的区切り）ベースのチャンキングを
  本命候補として、複数戦略を並列比較できるようにする
