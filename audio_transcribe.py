import re
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions

class AudioTranscriber:
    """
    音声ファイルからテキストを書き起こすためのクラス。
    Whisperモデルを使用して音声認識を行います。
    """
    def __init__(self):
        """
        AudioTranscriberクラスのコンストラクタ。
        Whisperモデルを初期化します。
        """
        # "small"モデルを使用し、GPU (cuda) でfloat16の計算タイプを使用します。
        self.model = WhisperModel("small", device="cuda", compute_type="float16")

    def transcribe(self, wav_path):
        """
        指定されたWAV音声ファイルをテキストに書き起こします。

        Args:
            wav_path (str): 書き起こしを行うWAVファイルのパス。

        Returns:
            list: 各セグメントの開始時刻、終了時刻、正規化されたテキストを含むリスト。
                  例: [[開始時刻1, 終了時刻1, "テキスト1"], [開始時刻2, 終了時刻2, "テキスト2"], ...]
        """
        # VADパラメータ設定（最新APIに合わせて修正）
        vad_parameters = VadOptions(
            min_speech_duration_ms=100,  # 最小の音声継続時間（ミリ秒）
            speech_pad_ms=100,           # 音声区間の前後に追加するパディング（ミリ秒）
            threshold=0.25,              # 音声と判断するための閾値
            # neg_thresholdパラメータは新しいバージョンでは削除されました
        )
        
        # 音声ファイルを書き起こします。
        segments, info = self.model.transcribe(
            wav_path,
            language="en",  # 英語として設定
            beam_size=5,
            vad_filter=True,
            vad_parameters=vad_parameters
        )
        
        # 書き起こし結果を収集します。
        results = []
        for seg in segments:
            # 英語のテキストに対して正規化処理を適用します。
            normalized_text = self.normalize_english_text(seg.text)
            results.append([
                seg.start,         # セグメントの開始時刻
                seg.end,           # セグメントの終了時刻
                normalized_text    # 正規化されたテキスト
            ])
        return results

    def normalize_english_text(self, text):
        """
        書き起こされた英語のテキストを正規化します。
        具体的には、最初の文字を大文字にする、句読点を修正する、一般的な間違いを修正する、
        不要なスペースを削除するなどの処理を行います。

        Args:
            text (str): 正規化する元のテキスト。

        Returns:
            str: 正規化されたテキスト。
        """
        # 最初の文字を大文字にします。
        if text:
            text = text[0].upper() + text[1:]
        
        # テキストの最後に適切な句読点 (., ?, !, :) がない場合、ピリオドを追加します。
        if text and not text.endswith(('.', '?', '!', ':')):
            text += '.'
        
        # 一般的な間違いを正規表現で置換します。
        # 単独の 'i' を 'I' に置換します。
        text = re.sub(r"\bi\b", "I", text)  
        # 短縮形 ('re, 's, 't) を展開します。例: "you're" -> "you are"
        text = re.sub(r"(\w)'re\b", r"\1 are", text)  # 例: "they're" -> "they are"
        text = re.sub(r"(\w)'s\b", r"\1 is", text)    # 例: "it's" -> "it is"
        text = re.sub(r"(\w)'t\b", r"\1 not", text)   # 例: "don't" -> "do not"
        
        # 不要なスペースを削除します。
        # 句読点の前のスペースを削除します。例: "Hello ." -> "Hello."
        text = re.sub(r"\s+([.,?!])", r"\1", text)  
        # 複数のスペースを1つのスペースに変換し、前後のスペースを削除します。
        text = re.sub(r"\s+", " ", text).strip()  
        
        return text
