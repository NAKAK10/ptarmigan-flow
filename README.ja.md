# ptarmigan-flow

macOS向けの Push-to-talk 文字起こしデーモンです。
グローバルホットキーを押している間だけ録音し、離したら Moonshine で文字起こししてアクティブアプリへ貼り付けます。
音声認識には公式リポジトリの Moonshine を利用しています: https://github.com/moonshine-ai/moonshine

[English](./README.md)

詳細な運用メモ、トラブルシュート、完全なコマンド/設定リファレンスは [README.llm.md](./README.llm.md) を参照してください。

## For LLM Agents
インストールガイドを取得して、それに従ってください:

```bash
curl -s https://raw.githubusercontent.com/NAKAK10/ptarmigan-flow/main/README.llm.md
```

## クイックスタート
```bash
brew install ptarmigan-flow
ptarmigan-flow doctor
ptarmigan-flow check-permissions --request
ptarmigan-flow run
```
`pflow` は短縮エイリアスなので、`pflow doctor` / `pflow run` も同じように使えます。

録音中に音楽再生の音質が落ちる場合（Bluetoothヘッドセット利用時など）は、
`audio.input_device` を未設定のまま `audio.input_device_policy = "playback_friendly"` を使ってください。

## 利用サンプル動画

<video src="./assets/usage-sample.mov" controls muted playsinline>
  お使いのブラウザは video タグに対応していません。
</video>

## コマンド一覧
### 主要コマンド
| コマンド | 説明 |
| --- | --- |
| `ptarmigan-flow init` | 現在値をデフォルトとして `config.toml` を対話的に編集します。 |
| `ptarmigan-flow run` | バックグラウンドデーモンを起動します。 |
| `ptarmigan-flow install-launch-agent` | 初回セットアップ用: launchd エージェントをインストールします。 |
| `ptarmigan-flow restart-launch-agent` | 新しく許可した macOS 権限を反映するために launchd エージェントを再起動します。 |

コマンドの全一覧とオプション:
- `ptarmigan-flow --help`
- `ptarmigan-flow <command> --help`

## インストール（Homebrew）
### 最短（推奨）
```bash
brew install ptarmigan-flow
```

### 任意ヘルパー（旧名移行 + リトライ付き）
```bash
./scripts/install_brew.sh
```

更新・削除:
```bash
brew upgrade ptarmigan-flow
brew uninstall ptarmigan-flow
```
