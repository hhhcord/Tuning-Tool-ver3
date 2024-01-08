import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import welch, csd
from scipy.fft import fft
from scipy.signal import butter, lfilter
import control as ctrl
from control.matlab import * 

# 信号処理を行うクラス
class SignalProcessor:
    def __init__(self, input_data, output_data, fs, cutoff_freq, time):
        self.input_data = input_data  # 入力データ
        self.output_data = output_data  # 出力データ
        self.fs = fs  # サンプリング周波数
        self.cutoff_freq = cutoff_freq # カットオフ周波数 (Hz)
        self.time = time # 読み取った録音データの時間

    def calculate_differences(self, time_series):
        """
        Calculate the differences in a time series data.
        The first element of the difference is set to 0 to keep the size same as the input.

        :param time_series: List of time series data
        :return: List of differences with the same size as time_series
        """
        if not time_series.size:
            return []

        # Initialize the difference list with the first element as 0
        differences = [0]

        # Calculate differences
        for i in range(1, len(time_series)):
            differences.append(time_series[i] - time_series[i - 1])

        return differences

    def identify_system(self):
        """
        システム同定
        """
        # 目標特性の定義
        A_des = '1 1; -0.5 -0.5'
        B_des = '0; 1'
        C_des1 = '1 0'
        C_des2 = '0 1'
        D_des = '0'
        ts = 1/self.fs # サンプリング時間

        # H_desの計算
        H_des1 = ctrl.StateSpace(A_des, B_des, C_des1, D_des)
        H_des2 = ctrl.StateSpace(A_des, B_des, C_des2, D_des)

        # FRITによる更新を行うために必要なデータの生成
        T = np.arange(0, self.time, ts)
        N = len(T)  # データの長さ
        n = 2   # 状態変数の数
        u_ini = self.input_data
        x_ini = np.zeros((N, n))
        x_ini[:, 0] = self.output_data
        x_ini[:, 1] = self.calculate_differences(self.output_data) # 差分のデータ

        # ΓとWの計算
        Gamma = np.zeros((n*N, 1))
        W = np.zeros((n*N, n))

        index = np.zeros((N, 1))
        gamma = np.zeros((N, 1))
        # 1の処理
        index, _, _ = ctrl.matlab.lsim(H_des1, u_ini, T)
        gamma = x_ini[:, 0] - index
        Gamma[0 * N:(0 + 1) * N, 0] = gamma.T
        index, _, _ = ctrl.matlab.lsim(H_des1, x_ini[:, 0], T)
        W[0 * N:(0 + 1) * N, 0] = index.T
        index, _, _ = ctrl.matlab.lsim(H_des1, x_ini[:, 1], T)
        W[0 * N:(0 + 1) * N, 1] = index.T
        # 2の処理
        index, _, _ = ctrl.matlab.lsim(H_des2, u_ini, T)
        gamma = x_ini[:, 1] - index
        Gamma[1 * N:(1 + 1) * N, 0] = gamma.T
        index, _, _ = ctrl.matlab.lsim(H_des2, x_ini[:, 0], T)
        W[1 * N:(1 + 1) * N, 0] = index.T
        index, _, _ = ctrl.matlab.lsim(H_des2, x_ini[:, 1], T)
        W[1 * N:(1 + 1) * N, 1] = index.T

        # F^*の計算
        F_ast = -np.dot(Gamma.T, np.dot(W, np.linalg.inv(np.dot(W.T, W))))
        print(F_ast)

        # システム同定
        A_hat = H_des1.A - np.dot(H_des1.B, F_ast)
        P = ctrl.StateSpace(A_hat, B_des, C_des1, D_des)
        print('同定されたシステム:')
        print(P)

        # 加工したデータの出力
        processed_data, _, _ = ctrl.matlab.lsim(P, u_ini, T)

        return processed_data

    def plot_fft_spectrum(self, signal, sampling_rate=2000):
        # FFTスペクトルをプロットする関数
        N = len(signal)  # サンプル点の数

        # FFTと周波数領域
        yf = fft(signal)
        xf = np.linspace(0.0, sampling_rate/2, N//2)

        # プロット
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.title("FFT Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def apply_low_pass_filter(self, signal, cutoff_frequency, sampling_rate=2000):
        # ローパスフィルタを適用する関数
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_frequency / nyquist

        # Butterworthフィルタを設計
        b, a = butter(N=1, Wn=normal_cutoff, btype='low', analog=False)

        # フィルタを適用
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def compute_power_spectrum(self, data):
        # パワースペクトルを計算する関数
        f, Pxx = welch(data, fs=self.fs)
        return f, Pxx

    def compute_cross_spectrum(self, input, output):
        # クロススペクトルを計算する関数
        f, Pxy = csd(input, output, fs=self.fs)
        return f, Pxy

    def compute_frequency_response_function(self, output):
        # 周波数応答関数を計算する関数
        f, Pxx = self.compute_power_spectrum(self.input_data)
        _, Pxy = self.compute_cross_spectrum(self.input_data, self.output_data)
        H = Pxy / Pxx
        _, Pxx = self.compute_power_spectrum(self.input_data)
        _, Pxy = self.compute_cross_spectrum(self.input_data, output)
        P = Pxy / Pxx
        return f, H, P

    def plot_bode(self, f, H, P):
        # ボード線図（ゲインと位相のグラフ）を描画する関数
        gain_h = 20 * np.log10(np.abs(H))  # ゲインをデシベル単位で計算
        phase_h = np.angle(H, deg=True)  # 位相を度単位で計算

        self.plot_fft_spectrum(gain_h)
        self.plot_fft_spectrum(phase_h)

        gain_h = self.apply_low_pass_filter(gain_h, self.cutoff_freq)
        phase_h = self.apply_low_pass_filter(phase_h, self.cutoff_freq)

        # P について
        gain_p = 20 * np.log10(np.abs(P))  # ゲインをデシベル単位で計算
        phase_p = np.angle(H, deg=True)  # 位相を度単位で計算

        self.plot_fft_spectrum(gain_p)
        self.plot_fft_spectrum(phase_p)

        gain_p = self.apply_low_pass_filter(gain_p, self.cutoff_freq)
        phase_p = self.apply_low_pass_filter(phase_p, self.cutoff_freq)

        # 差の計算
        gain = gain_h - gain_p
        phase = phase_h - phase_p

        # Hのボード線図を表示したい場合
        # gain = gain_h
        # phase = phase_h

        # 同定したシステムのボード線図を表示したい場合
        # gain = gain_p
        # phase = phase_p

        # カスタムフォーマッタの定義
        def custom_formatter(x, pos):
            if x >= 1000:
                return '{:.0f}k'.format(x / 1000)
            else:
                return '{:.0f}'.format(x)

        # ゲイン線図を描画
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.semilogx(f, gain, base=2)
        plt.title('Bode Plot')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        # 位相線図を描画
        plt.subplot(2, 1, 2)
        plt.semilogx(f, phase, base=2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [degrees]')
        plt.xlim(20, 20000)
        plt.grid(which='both', linestyle='-', linewidth='0.5')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        plt.gca().xaxis.set_minor_formatter(ticker.FuncFormatter(custom_formatter))

        plt.tight_layout()
        plt.show()

    def find_flat_gain(self, f, H, P):
        # 平坦な部分のゲインを見つける関数
        gain_h = 20 * np.log10(np.abs(H))
        gain_h = self.apply_low_pass_filter(gain_h, self.cutoff_freq)
        # P について
        gain_p = 20 * np.log10(np.abs(P))
        gain_p = self.apply_low_pass_filter(gain_p, self.cutoff_freq)
        # 差の計算
        gain = gain_h - gain_p
        # 平坦な部分を特定するための簡易アルゴリズム（例えば標準偏差が小さい区間）
        std_gain = np.convolve(gain, np.ones(10)/10, mode='valid')  # 移動平均に基づく標準偏差
        flat_idx = np.argmin(np.abs(std_gain))  # 最も平坦な部分
        return f[flat_idx], gain[flat_idx]

    def find_peak_gains(self, f, H, P):
        # 平坦な部分のゲインを見つける関数
        gain_h = 20 * np.log10(np.abs(H))
        gain_h = self.apply_low_pass_filter(gain_h, self.cutoff_freq)
        # P について
        gain_p = 20 * np.log10(np.abs(P))
        gain_p = self.apply_low_pass_filter(gain_p, self.cutoff_freq)
        # 差の計算
        gain = gain_h - gain_p
        peaks, _ = find_peaks(gain, height=0)  # 山の頂点を見つける
        # ゲインが高い順に並べ替え
        peak_gains = sorted(zip(f[peaks], gain[peaks]), key=lambda x: x[1], reverse=True)
        return peak_gains[:3]  # 最大3つの山

    def gain_difference(self, f, H, P):
        # 平坦な部分と山の頂点のゲインの差を計算する関数
        _, flat_gain = self.find_flat_gain(f, H, P)
        peak_gains = self.find_peak_gains(f, H, P)
        differences = [(pf, pg - flat_gain) for pf, pg in peak_gains]
        return differences

    def write_results_to_file(self, filepath, f, H, P):
        with open(filepath, 'w') as file:
            # 平坦なゲインの値を書き出す
            flat_freq, flat_gain = self.find_flat_gain(f, H, P)
            file.write(f"平坦なゲインの周波数: {flat_freq:.2f} Hz, ゲイン: {flat_gain:.2f} dB\n")

            # 山の頂点のゲインの値を書き出す
            peak_gains = self.find_peak_gains(f, H, P)
            file.write("山の頂点のゲインの値（上位3つ）:\n")
            for freq, gain in peak_gains:
                file.write(f"  周波数: {freq:.2f} Hz, ゲイン: {gain:.2f} dB\n")

            # ゲインの差を書き出す
            gain_diff = self.gain_difference(f, H, P)
            file.write("ゲインの差（平坦な部分と山の頂点）:\n")
            for freq, diff in gain_diff:
                file.write(f"  周波数: {freq:.2f} Hz, ゲインの差: {diff:.2f} dB\n")

'''
# サンプルデータでの使用例（実際のオーディオデータに置き換えてください）
input_data = np.random.randn(1000)  # ダミー入力データ
output_data = np.random.randn(1000)  # ダミー出力データ
sampling_rate = 48000  # サンプリングレートの例

# SignalProcessorのインスタンスを作成
signal_processor = SignalProcessor(input_data, output_data, sampling_rate)

# 入力のパワースペクトルを計算
f, power_spectrum = signal_processor.compute_power_spectrum(input_data)

# 入力と出力のクロススペクトルを計算
f, cross_spectrum = signal_processor.compute_cross_spectrum()

# 周波数応答関数を計算
f, frequency_response = signal_processor.compute_frequency_response_function()

# デモンストレーション用に最初の数値を表示
(f[:5], power_spectrum[:5]), (f[:5], cross_spectrum[:5]), (f[:5], frequency_response[:5])

# ボード線図を描画
signal_processor.plot_bode(f, frequency_response)

# 平坦なゲインの値を取得
flat_freq, flat_gain = signal_processor.find_flat_gain(f, frequency_response)

# 山の頂点のゲインの値を取得
peak_gains = signal_processor.find_peak_gains(f, frequency_response)

# ゲインの差を取得
gain_diff = signal_processor.gain_difference(f, frequency_response)

# 例としてダミー信号を使用します
# デモンストレーション用のダミー信号を作成します
np.random.seed(0)  # 乱数のシードを0に設定して、結果を再現可能にします
dummy_signal = np.random.normal(0, 1, 2000)  # 平均0、標準偏差1の正規分布に従う2000個の乱数を生成して、例の信号とします

# ダミー信号のFFTスペクトラムをプロットします
plot_fft_spectrum(dummy_signal)  # FFTスペクトラムを表示する関数を使用

# 例としてダミー信号とカットオフ周波数を使用します
cutoff_freq = 500  # カットオフ周波数をHz単位で設定します

# ダミー信号にLPF（ローパスフィルター）を適用します
filtered_signal = apply_low_pass_filter(dummy_signal, cutoff_freq)  # LPFを適用した信号を取得

# 元の信号とフィルター処理された信号を比較してプロットします
plt.figure(figsize=(12, 6))  # 図のサイズを設定
plt.plot(dummy_signal, label="Original Signal")  # 元の信号をプロット
plt.plot(filtered_signal, label="Filtered Signal", alpha=0.7)  # フィルター処理された信号をプロット、透明度を0.7に設定
plt.title("Original vs. Low-Pass Filtered Signal")  # タイトルを設定
plt.xlabel("Sample Number")  # X軸のラベルを設定
plt.ylabel("Amplitude")  # Y軸のラベルを設定
plt.legend()  # 凡例を表示
plt.grid()  # グリッド線を表示
plt.show()  # 図を表示
'''
