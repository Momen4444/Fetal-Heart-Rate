import numpy as np
import pandas as pd
from scipy import signal, interpolate, ndimage
from scipy.integrate import trapezoid

class SignalProcessor:
    """
    Handles all mathematical operations, filtering, and metric extraction.
    """
    
    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.filtfilt(b, a, data)
        return y

    @staticmethod
    def notch_filter(data, cutoff, fs, Q=30):
        nyq = 0.5 * fs
        freq = cutoff / nyq
        b, a = signal.iirnotch(freq, Q)
        y = signal.filtfilt(b, a, data)
        return y

    @staticmethod
    def pan_tompkins_detector(ecg_signal, fs):
        """
        Simplified Pan-Tompkins Algorithm for QRS Detection.
        """
        # 1. Bandpass Filter (5-15 Hz)
        filtered = SignalProcessor.butter_bandpass_filter(ecg_signal, 5, 15, fs, order=1)
        
        # 2. Derivative
        derivative = np.diff(filtered)
        
        # 3. Squaring
        squared = derivative ** 2
        
        # 4. Moving Window Integration
        window_size = int(0.150 * fs) # 150ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 5. Peak Detection
        # Use adaptive thresholding or simple height check on integrated signal
        # Here we use scipy find_peaks on the integrated signal for robustness
        peaks, _ = signal.find_peaks(integrated, distance=int(fs*0.4), height=np.mean(integrated)*1.5)
        
        # Refine peaks: find max in original filtered signal near the integrated peaks
        refined_peaks = []
        search_window = int(fs * 0.1)
        for p in peaks:
            start = max(0, p - search_window)
            end = min(len(filtered), p + search_window)
            if end > start:
                local_max = np.argmax(filtered[start:end]) + start
                refined_peaks.append(local_max)
                
        return np.array(refined_peaks)

    @staticmethod
    def sample_entropy(U, m=2, r=0.2):
        """
        Calculate Sample Entropy (SampEn) of a time series.
        """
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
            return sum(C)

        N = len(U)
        # Normalize r by std if not already done, usually r = 0.2 * std
        r = r * np.std(U)
        
        return -np.log(_phi(m+1) / _phi(m)) if _phi(m) > 0 and _phi(m+1) > 0 else 0

    @staticmethod
    def preprocess_ecg(data, fs=250):
        """
        Applies Bandpass and Notch filtering for ECG/Signal cleaning.
        """
        # 1. Remove Power-line Noise (50Hz & 60Hz)
        data = SignalProcessor.notch_filter(data, 50.0, fs)
        data = SignalProcessor.notch_filter(data, 60.0, fs)
        
        # 2. Remove Baseline Wander (High-pass > 0.5Hz) & High Freq Noise (Low-pass < 40Hz)
        # Using Bandpass 0.5 - 40 Hz
        data = SignalProcessor.butter_bandpass_filter(data, 0.5, 40.0, fs, order=2)
        
        return data

    @staticmethod
    def calculate_hrv_metrics(rr_intervals_ms):
        """
        Advanced HRV Analysis: Time, Frequency, and Non-linear domains.
        """
        # --- 1. Filtering (Outlier Removal) ---
        # Remove beats that deviate > 20% from median (ectopic beat filtering)
        median_rr = np.median(rr_intervals_ms)
        valid_indices = np.abs(rr_intervals_ms - median_rr) < 0.2 * median_rr
        rr_clean = rr_intervals_ms[valid_indices]
        
        if len(rr_clean) < 10: return None # Not enough data after filtering

        # --- 2. Time Domain ---
        rr_diff = np.diff(rr_clean)
        sdnn = np.std(rr_clean, ddof=1)
        rmssd = np.sqrt(np.mean(rr_diff ** 2))
        mean_hr = 60000 / np.mean(rr_clean)
        
        # pNN50 & NN50
        nn50 = np.sum(np.abs(rr_diff) > 50)
        pnn50 = (nn50 / len(rr_diff)) * 100 if len(rr_diff) > 0 else 0
        
        # Geometric: HRV Triangular Index (Total beats / Height of histogram)
        hist, _ = np.histogram(rr_clean, bins=range(int(min(rr_clean)), int(max(rr_clean)) + 8, 8)) # 8ms bins
        hrv_triangular_idx = len(rr_clean) / max(hist) if max(hist) > 0 else 0

        # --- 3. Frequency Domain (Welch's method) ---
        # Resample to 4Hz
        x_time = np.cumsum(rr_clean) / 1000.0
        x_time = x_time - x_time[0]
        
        f_interp = interpolate.interp1d(x_time, rr_clean, kind='cubic', fill_value="extrapolate")
        fs = 4.0
        time_steps = np.arange(0, x_time[-1], 1/fs)
        rr_interpolated = f_interp(time_steps)
        rr_detrended = signal.detrend(rr_interpolated)
        
        freqs, psd = signal.welch(rr_detrended, fs=fs, nperseg=256)
        
        # Band Integration
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.40)
        
        vlf_power = trapezoid(psd[(freqs >= vlf_band[0]) & (freqs <= vlf_band[1])], freqs[(freqs >= vlf_band[0]) & (freqs <= vlf_band[1])])
        lf_power = trapezoid(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
        hf_power = trapezoid(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        
        # --- 4. Non-linear Domain ---
        # Poincare Plot
        x1 = rr_clean[:-1]
        x2 = rr_clean[1:]
        sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
        sd2 = np.std(np.add(x1, x2) / np.sqrt(2))
        
        # Sample Entropy (Approximate calculation for speed on short segments)
        # Using a subset if data is too long to prevent UI freeze
        samp_en = SignalProcessor.sample_entropy(rr_clean[:500]) if len(rr_clean) > 0 else 0

        return {
            'SDNN': sdnn, 'RMSSD': rmssd, 'Mean HR': mean_hr, 'pNN50': pnn50,
            'HRV_Tri_Idx': hrv_triangular_idx,
            'VLF Power': vlf_power, 'LF Power': lf_power, 'HF Power': hf_power, 'LF/HF Ratio': lf_hf_ratio,
            'SD1': sd1, 'SD2': sd2, 'SampEn': samp_en,
            'PSD_Freqs': freqs, 'PSD_Power': psd,
            'RR_Interp': rr_interpolated, 'Time_Axis': time_steps
        }

    @staticmethod
    def calculate_ctg_metrics(fhr_signal, fs=4):
        """
        Advanced CTG Analysis: Baseline, Variability (STV/LTV), Acc/Dec detection.
        """
        # 1. Preprocessing (Artifact Removal already done in preprocess_signal, but double check)
        clean_fhr = fhr_signal[(fhr_signal > 50) & (fhr_signal < 220)]
        if len(clean_fhr) == 0: return None

        # 2. Baseline Estimation (Mantel et al. approach simplified)
        # Low-pass filter to remove variability and accelerations
        b, a = signal.butter(2, 0.01, btype='low', fs=fs) # Very low cutoff
        baseline_trend = signal.filtfilt(b, a, clean_fhr)
        baseline = np.mean(baseline_trend) # Use mean of trend as stable baseline

        # 3. Variability Analysis
        # STV: Mean absolute difference of beat-to-beat (or epoch-to-epoch)
        stv = np.mean(np.abs(np.diff(clean_fhr)))
        
        # LTV: Long Term Variability (Range within 1-minute windows)
        # Window size = 1 min = 60 * fs samples
        window_size = 60 * fs
        ltv_list = []
        for i in range(0, len(clean_fhr) - window_size, window_size):
            segment = clean_fhr[i:i+window_size]
            ltv_list.append(np.max(segment) - np.min(segment))
        ltv = np.mean(ltv_list) if ltv_list else 0

        # 4. Accelerations & Decelerations
        # Acceleration: >15bpm for >15s
        # Deceleration: <15bpm for >15s
        deviation = clean_fhr - baseline
        
        # Identify segments
        is_acc = deviation > 15
        is_dec = deviation < -15
        
        # Count duration (consecutive samples)
        def count_events(bool_array, min_duration_sec):
            min_samples = min_duration_sec * fs
            labeled, num_features = ndimage.label(bool_array)
            count = 0
            total_time = 0
            for i in range(1, num_features + 1):
                duration = np.sum(labeled == i)
                if duration >= min_samples:
                    count += 1
                    total_time += duration
            return count, total_time / fs

        num_acc, time_acc = count_events(is_acc, 15)
        num_dec, time_dec = count_events(is_dec, 15)
        
        # 5. Sinusoidal Pattern Detection (Sign of severe distress)
        # Check for regular oscillation in 3-5 cycles/min frequency range
        freqs, psd = signal.welch(clean_fhr, fs=fs, nperseg=512)
        # 3-5 cycles/min = 0.05 - 0.08 Hz
        sinusoidal_power = trapezoid(psd[(freqs >= 0.05) & (freqs <= 0.08)], freqs[(freqs >= 0.05) & (freqs <= 0.08)])
        total_power = trapezoid(psd, freqs)
        sinusoidal_index = sinusoidal_power / total_power if total_power > 0 else 0

        # Statistical Features for ML Model
        mean_f = np.mean(clean_fhr)
        median_f = np.median(clean_fhr)
        std_f = np.std(clean_fhr)
        peak_f = np.max(clean_fhr)
        
        ml_features = pd.DataFrame([{
            'Mean': mean_f,
            'Median': median_f,
            'Std': std_f,
            'Max': peak_f
        }])
        
        return {
            'Baseline': baseline, 
            'STV': stv, 'LTV': ltv,
            'Acc_Count': num_acc, 'Acc_Time': time_acc,
            'Dec_Count': num_dec, 'Dec_Time': time_dec,
            'Sinusoidal_Idx': sinusoidal_index,
            'Clean_Signal': clean_fhr,
            'ML_Features': ml_features
        }
