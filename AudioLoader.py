import tkinter as tk  # tkinterモジュールをインポート（GUI関連の機能）
from tkinter import filedialog  # ファイル選択ダイアログ関連の機能をインポート
import soundfile as sf  # soundfileモジュールをインポート（オーディオファイルの読み込み・書き込みに使用）

class AudioLoader:
    def __init__(self):
        # オーディオファイルの形式を指定
        self.file_types = [('Audio Files', '*.wav *.mp3 *.m4a')]

    # load_audioメソッドに読み込み秒数のパラメータを追加
    def load_audio(self, seconds=5):
        root = tk.Tk()  # Tkinterのルートウィンドウを作成
        root.withdraw()  # ルートウィンドウを非表示にする
        file_path = filedialog.askopenfilename(filetypes=self.file_types)  # ファイル選択ダイアログを表示し、ファイルパスを取得
        if file_path:
            print(f"User selected {file_path}")  # ユーザーが選んだファイルパスを表示
            # 最初にファイルを開いてサンプリングレートを取得
            data, fs = sf.read(file_path, dtype='float32')
            # ユーザーが指定した秒数の長さのデータを再読込
            data, fs = sf.read(file_path, start=0, stop=seconds*fs, dtype='float32')
            return data, fs
        else:
            print("User selected Cancel")  # ユーザーがキャンセルした場合
            return None, None  # Noneを返す

'''
# 使用例
audio_loader = AudioLoader()
audio_data, sampling_rate = audio_loader.load_audio(10)  # 10秒のオーディオを読み込む
'''
