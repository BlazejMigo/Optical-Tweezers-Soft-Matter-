import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AutocorrelationAnalyzer:
    """
    autocorrelation of tracking data from multiple CSV files
    """
    
    def __init__(self, pattern="Tracking_output_*.csv", fps=650.0, 
                 header_row=3, axis_col="POSITION_X", remove_trend=True):
        """
        Initialize the analyzer with configuration parameters.
        """
        self.pattern = pattern
        self.fps = fps
        self.header_row = header_row
        self.axis_col = axis_col
        self.remove_trend = remove_trend
        self.files = []
        self.all_autocorrelations = []
        self.tau = None
        self.A_mean = None
        self.tauc = None
        
    def load_files(self):
        """
        Load CSV files matching the pattern
        """
        self.files = sorted(glob.glob(self.pattern))
        if not self.files:
            raise FileNotFoundError(f"No files found matching pattern {self.pattern!r}")
        
    def _autocorr_fft(self, x):
        """
        Compute normalized autocorrelation using FFT.
        """
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
        n = len(x)
        
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([1.0], dtype=float)
        
        nfft = 1 << (2 * n - 1).bit_length()
        f = np.fft.rfft(x, n=nfft)
        ac = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
        denom = ac[0] if ac.size > 0 and ac[0] != 0.0 else 1.0
        return ac / denom
    
    def _detect_column(self, df, fname):
        """
        Detect appropriate column if the specified one doesn't exist
        """
        if self.axis_col in df.columns:
            return self.axis_col
            
        candidates = [
            c for c in df.columns
            if ('pixel' in str(c).lower() or 'position' in str(c).lower() or 
                str(c).strip().lower().endswith('x') or str(c).strip().lower() == 'x')
        ]
        
        if candidates:
            return candidates[0]
        
        raise ValueError(f"Column {self.axis_col!r} not found in {fname}")
    
    def _detrend(self, x):
        """
        Remove linear trend from data
        """
        t = np.arange(len(x)) / self.fps
        coeffs = np.polyfit(t, x, 1)
        trend = np.polyval(coeffs, t)
        return x - trend
    
    def process_files(self):
        """
        Process all CSV files and compute autocorrelations
        """
        for fname in self.files:
            df = pd.read_csv(fname, header=self.header_row)
            
            try:
                col_name = self._detect_column(df, fname)
            except ValueError:
                continue
            
            x = pd.to_numeric(df[col_name], errors="coerce").to_numpy()
            x = x[np.isfinite(x)]
            
            if x.size < 2:
                continue
            
            if self.remove_trend:
                x = self._detrend(x)
            
            A = self._autocorr_fft(x)
            if A.size > 0:
                self.all_autocorrelations.append(A)
        
        if not self.all_autocorrelations:
            raise RuntimeError("No valid autocorrelations computed")
    
    def compute_average(self):
        """
        Compute average autocorrelation from all files.
        """
        Nmin = min(len(A) for A in self.all_autocorrelations)
        all_A_trunc = [A[:Nmin] for A in self.all_autocorrelations]
        self.A_mean = np.mean(all_A_trunc, axis=0)
        self.tau = np.arange(Nmin) / self.fps
    
    def fit_exponential(self):
        """
        Fit single exponential decay to averaged autocorrelation.
        """
        tau = self.tau[self.tau > 0]
        A = self.A_mean[self.tau > 0]
        
        mask = (A > 0.1) & (A < 0.9) & np.isfinite(A)
        
        if np.count_nonzero(mask) < 5:
            raise RuntimeError("Not enough points to fit exponential")
        
        lnA = np.log(A[mask])
        m, _ = np.polyfit(tau[mask], lnA, 1)
        
        if m >= 0:
            raise RuntimeError("Fitted slope is non-negative")
        
        self.tauc = -1.0 / m
        return self.tauc
    
    def save_results(self, output_file="average_autocorrelation.csv"):
        """
        Save averaged autocorrelation to CSV.
        """
        out_df = pd.DataFrame({"tau_s": self.tau, "A_tau": self.A_mean})
        out_df.to_csv(output_file, index=False)
    
    def plot_results(self):
        """
        Plot averaged autocorrelation with optional theoretical fit.
        """
        plt.figure()
        
        if len(self.tau) <= 1:
            plt.plot(self.tau, self.A_mean, linewidth=2.0, label="Average data")
            if self.tauc is not None:
                A_theory = np.exp(-self.tau / self.tauc)
                plt.plot(self.tau, A_theory, linestyle="--", linewidth=1.5,
                        label=r"Theory: $e^{-\tau/\tau_c}$")
        else:
            tau_plot = self.tau[1:]
            A_mean_plot = self.A_mean[1:]
            plt.semilogx(tau_plot, A_mean_plot, linewidth=2.0, color='black', 
                        label="Average data")
            
            if self.tauc is not None:
                A_theory_plot = np.exp(-tau_plot / self.tauc)
                plt.semilogx(tau_plot, A_theory_plot, linestyle="--", linewidth=1.5,
                           label=rf"Theory: $e^{{-\tau/\tau_c}}$, $\tau_c={self.tauc:.2e}\,\mathrm{{s}}$")
        
        plt.xlabel(r"$\tau$ [s]")
        plt.ylabel(r"$A(\tau)$")
        plt.ylim(-0.1, 1.05)
        plt.grid(True, which="both", linestyle=":", linewidth=0.5)
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """
        Run complete analysis pipeline.
        """
        self.load_files()
        self.process_files()
        self.compute_average()
        
        try:
            self.fit_exponential()
        except RuntimeError:
            pass
        
        self.save_results()
        self.plot_results()


def main():
    """
    Main function to run autocorrelation analysis.
    """
    analyzer = AutocorrelationAnalyzer(
        pattern="Tracking_output_*.csv",
        fps=650.0,
        header_row=3,
        axis_col="POSITION_X",
        remove_trend=True
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
