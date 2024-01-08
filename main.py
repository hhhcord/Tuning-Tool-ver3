from AudioLoader import AudioLoader  # AudioLoaderクラスをインポート
from SignalProcessor import SignalProcessor  # SignalProcessorクラスをインポート

# AudioLoaderのインスタンスを作成
audio_loader = AudioLoader()

time = 30 # 読み取る秒数の指定

# load_audioメソッドを使用して、30秒間の入力オーディオデータを読み込む
input_data, sampling_rate = audio_loader.load_audio(time)

# 同様に、30秒間の出力オーディオデータを読み込む
output_data, sampling_rate = audio_loader.load_audio(time)

cutoff_freq = 80  # カットオフ周波数 (Hz)

print('進捗(1/4)')

# SignalProcessorのインスタンスを作成。入力データ、出力データ、サンプリングレートを渡す
signal_processor = SignalProcessor(input_data, output_data, sampling_rate, cutoff_freq, time)

# 加工したデータの出力
processed_data = signal_processor.identify_system()

print('進捗(2/4)')

# 周波数応答関数を計算する
f, H, P = signal_processor.compute_frequency_response_function(processed_data)
# H: 計測データでの応答
# P: 同定したシステムでの応答

print('進捗(3/4)')

# 結果をテキストファイルに書き出すためのメソッドを呼び出す
# "results.txt"には、計算された周波数応答関数のデータが保存される
signal_processor.write_results_to_file("results.txt", f, H, P)

# ボード線図を描画する
# ボード線図は、システムの周波数応答を示すグラフで、ゲインと位相の変化を視覚的に表示する
signal_processor.plot_bode(f, H, P)

print('進捗(4/4)')
