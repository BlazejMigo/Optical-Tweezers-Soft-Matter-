import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class OpticalConfig:
    """
    Configuration for experiment parameters.
    """
    
    def __init__(self):
        self.csv_path = "Tracking_output_1.csv"
        self.camera_pixel_um = 4.8
        self.objective_mag = 100.0
        self.binning = 1.0
        self.fps_override = 10000 / 15.6
        
    @property
    def px_size_um(self):
        """
        Calculate pixel size in micrometers.
        """
        return self.camera_pixel_um * self.binning / self.objective_mag


class TrackDataLoader:
    """
    handles loading and cleaning of Trackmate CSV data.
    """
    
    @staticmethod
    def load_trackmate_csv(csv_path):
        """
        Load and clean Trackmate CSV file
        """
        df = pd.read_csv(csv_path, encoding_errors="ignore")
        df = df.drop([0, 1, 2])  # Remove header/unit rows
        
        for col in ["POSITION_X", "POSITION_Y", "FRAME", "POSITION_T", "TRACK_ID"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna(subset=["POSITION_X", "POSITION_Y", "FRAME"])
        return df


class CoordinateTransformer:
    """
    Transforms pixel coordinates to physical units.
    """
    
    def __init__(self, config):
        self.config = config
    
    def frame_to_time(self, frames):
        """
        Convert frame numbers to time in seconds.
        """
        return frames / self.config.fps_override
    
    def pixels_to_micrometers(self, pixels):
        """
        Convert pixel positions to micrometers.
        """
        return pixels * self.config.px_size_um


class TrapAnalyzer:
    """
    Analyzes trap positions
    """
    
    def __init__(self, config):
        self.config = config
    
    def find_trap_centers(self, r_raw):
        """
        Find the two trap centers using Gaussian mixture model.
        """
        gmm = GaussianMixture(
            n_components=2, 
            covariance_type="diag", 
            random_state=0
        ).fit(r_raw.reshape(-1, 1))
        
        means = np.sort(gmm.means_.ravel())
        mid = means.mean()
        D0 = (means[1] - means[0]) / 2.0
        
        return mid, D0
    
    def calculate_signed_position(self, r_raw, midpoint):
        """
        Calculate signed position relative to trap midpoint
        """
        return r_raw - midpoint


class FlipVisualizer:
    """
    Creates publication-style visualizations of flip data
    """
    
    def __init__(self):
        self._setup_matplotlib()
    
    @staticmethod
    def _setup_matplotlib():
        """
        Configure matplotlib for publication-style plots
        """
        mpl.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        })
    
    def plot_flip_trace(self, t, r, D0, t_min=20.0, t_max=100.0, fake_span=90.0):
        """
        Create publication-style plot of flip trace
        """
        mask = (t >= t_min) & (t <= t_max)
        t_plot = t[mask]
        r_plot = r[mask]
        
        ylim = 1.3 * np.nanmax(np.abs(r_plot))
        ylim = max(ylim, 1.3 * abs(D0))
        
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        
        # Plot raw data
        ax.plot(
            t_plot, r_plot,
            linestyle="none",
            marker="o",
            markersize=2,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.6,
            alpha=0.9,
        )
        
        # Set limits and labels
        ax.set_ylim(-ylim, +ylim)
        ax.set_xlim(t_min, t_max)
        ax.set_ylabel(r"r [µm]")
        
        # Create artificial 0-90 s axis
        n_ticks = 7
        tick_positions = np.linspace(t_min, t_max, n_ticks)
        fake_labels = np.linspace(0.0, fake_span, n_ticks)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.0f}" for v in fake_labels])
        ax.set_xlabel("Time [s]")
        
        fig.tight_layout()
        plt.show()


def main():
    """
    Main function to run flip analysis
    """
    # Initialize configuration
    config = OpticalConfig()
    
    # Load data
    loader = TrackDataLoader()
    df = loader.load_trackmate_csv(config.csv_path)
    
    # Transform coordinates
    transformer = CoordinateTransformer(config)
    frame = df["FRAME"].to_numpy(dtype=float)
    t = transformer.frame_to_time(frame)
    
    x_um = transformer.pixels_to_micrometers(df["POSITION_X"].to_numpy(dtype=float))
    y_um = transformer.pixels_to_micrometers(df["POSITION_Y"].to_numpy(dtype=float))
    
    # Use X-axis as flip axis
    r_raw = x_um
    
    # Analyze traps
    analyzer = TrapAnalyzer(config)
    mid, D0 = analyzer.find_trap_centers(r_raw)
    r = analyzer.calculate_signed_position(r_raw, mid)
    
    # Visualize
    visualizer = FlipVisualizer()
    visualizer.plot_flip_trace(t, r, D0, t_min=20.0, t_max=100.0, fake_span=90.0)


if __name__ == "__main__":
    main()
