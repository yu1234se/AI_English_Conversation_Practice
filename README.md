# AI英会話練習アプリケーション - README
![app1A](https://github.com/user-attachments/assets/d3941774-e2bd-4948-a0f2-ff742a3e8b21)
## 概要
このプロジェクトはKokoro、Whisper、LangChain、Gemma 3を活用した英会話練習アプリケーションで、インタラクティブな言語学習体験を提供します。システムの主な機能は以下の通りです：
1. 英語で音声を録音
2. 音声を自動的にテキストに変換
3. AIが生成した自然な音声での応答を受け取り

## ファイル構成

### 1. `app.py`
メインのStreamlitアプリケーションで、ユーザーインターフェースとコアワークフローを処理します。

**主な機能:**
- 🎙️ カスタムUIによる音声録音機能
- 🔄 リアルタイム会話フロー管理
- 🔊 速度調整可能なテキスト音声変換
- 📝 音声再生付き会話履歴表示
- ⚙️ 音声速度調整用サイドバー設定

**依存ライブラリ:**
- `streamlit`（Webインターフェース）
- `sounddevice`/`soundfile`（音声録音）
- `numpy`（音声処理）
- `kokoro`（テキスト音声合成）

### 2. `audio_transcribe.py`
Whisperモデルを使用した音声認識処理を行います。

**主な機能:**
- 🔍 高精度な英語音声認識
- 🎚️ ノイズ除去のための音声活動検出(VAD)
- ⚡ GPU高速処理による素早い文字起こし

**処理ステップ:**
1. VADによる音声セグメンテーション
2. Whisperモデルによる音声認識
3. テキスト正規化：
   - 大文字化修正
   - 句読点調整
   - 短縮形展開（"don't" → "do not"）
   - 空白最適化

### 3. `conversation_agent.py`
文脈を考慮した英会話応答を生成します。

**主な機能:**
- 💬 英語講師としてのロールプレイ
- 🧠 LLMによる応答生成（Gemma 3モデル）
- 📚 会話コンテキスト管理




## システム要件
- Python 3.9以上
- NVIDIA GPU
- LM Studio導入済み


## インストール方法
```bash
# リポジトリをクローン
git clone https://github.com/your-username/ai-english-practice.git

# 依存ライブラリをインストール
pip install -r requirements.txt


# LM StudioでGemma 3モデルを起動
# http://localhost:1234 で実行するように設定
```

## 使用方法
1. アプリケーションを起動:
```bash
streamlit run app.py
```

2. ブラウザで操作:
   - 「🎤 Start Recording」をクリックして発声
   - 終了後「⏹️ Stop Recording」をクリック
   - 「Transcribe and Send」で音声処理
   - AIの音声応答を聴く
   - サイドバーのスライダーで音声速度を調整

## 設定オプション
- **音声速度**: 0.5倍～2.0倍（デフォルト:1.0倍）
- **会話履歴**: 直近6メッセージを保持

## トラブルシューティング
- **音声が録音されない**: マイクの権限を確認
- **文字起こしが遅い**: GPUの利用可否とCUDAインストールを確認
- **LLMが応答しない**: LM Studioが`http://localhost:1234`で実行中か確認
- **音質の問題**: 録音時の背景ノイズを軽減

## ライセンス
このプロジェクトはMITライセンスのもとで公開されています - [LICENSE](LICENSE) ファイルを参照してください。
