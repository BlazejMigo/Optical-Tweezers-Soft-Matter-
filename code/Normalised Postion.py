import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class FlipAnalysisConfig:
    """
    Configuration parameters for flip analysis.
    """
    
    def __init__(self):
        self.pattern = "Tracking_output_*.csv"
        self.pixel_size_um = 0.048
        self.fps = 650.0
        self.pre_time = 2
        self.post_time = 16.0
        self.start_win = 0.1
        self.end_win = 1.0
        self.output_csv = "flip_mean_trace.csv"


class FlipDataProcessor:
    """
    Handles processing of flip tracking data.
    """
    
    def __init__(self, config):
        self.config = config
        
    def load_and_clean_file(self, filepath):
        """
        Load a CSV file and clean the data.
        """
        df = pd.read_csv(filepath, skiprows=[1, 2], low_memory=False)
        t_raw = pd.to_numeric(df["POSITION_T"], errors="coerce").to_numpy()
        x_px = pd.to_numeric(df["POSITION_X"], errors="coerce").to_numpy()
        
        # Remove NaN values
        good = ~np.isnan(t_raw)
        t_raw = t_raw[good]
        x_px = x_px[good]
        
        # Sort by time
        order = np.argsort(t_raw)
        t_raw = t_raw[order]
        x_px = x_px[order]
        
        if len(t_raw) < 5:
            return None, None
            
        return t_raw, x_px
    
    def detect_flip_event(self, t_sec, x_um):
        """
        Detect the flip event as the biggest negative step.
        """
        dx = np.diff(x_um)
        step_idx = np.argmin(dx)
        return t_sec[step_idx]
    
    def extract_flip_segment(self, t_sec, x_um, t_step):
        """
        Extract data segment around the flip event.
        """
        mask = (t_sec >= t_step - self.config.pre_time) & \
               (t_sec <= t_step + self.config.post_time)
        t_seg = t_sec[mask] - t_step
        x_seg = x_um[mask]
        
        if len(t_seg) < 5:
            return None, None
            
        return t_seg, x_seg
    
    def process_file(self, filepath):
        """
        Process a single tracking file.
        """
        t_raw, x_px = self.load_and_clean_file(filepath)
        if t_raw is None:
            return None, None, None
        
        # Convert to physical units
        t_sec = t_raw / self.config.fps
        x_um = x_px * self.config.pixel_size_um
        
        # Detect flip and extract segment
        t_step = self.detect_flip_event(t_sec, x_um)
        t_seg, x_seg = self.extract_flip_segment(t_sec, x_um, t_step)
        
        if t_seg is None:
            return None, None, None
        
        dt = np.median(np.diff(t_seg))
        return t_seg, x_seg, dt


class FlipNormalizer:
    """
    Normalizes flip segments to a common scale.
    """
    
    def __init__(self, config):
        self.config = config
    
    def normalize_segment(self, t_seg, x_seg, t_end):
        """
        Normalize a single segment to range [0, 1].
        """
        # Start plateau: just before the flip
        start_mask = (t_seg <= 0) & (t_seg >= -self.config.start_win)
        x_start = x_seg[start_mask].mean()
        
        # Final plateau: last period of segment
        if t_seg[-1] > (t_end - self.config.end_win):
            end_mask = t_seg >= (t_seg[-1] - self.config.end_win)
        else:
            n_tail = max(3, int(0.2 * len(x_seg)))
            end_mask = np.zeros_like(t_seg, dtype=bool)
            end_mask[-n_tail:] = True
        x_final = x_seg[end_mask].mean()
        
        denom = x_start - x_final
        if abs(denom) < 1e-9:
            return None, None, None
        
        x_norm = (x_seg - x_final) / denom
        return x_start, x_final, x_norm


class FlipVisualizer:
    """
    Creates visualizations of flip data.
    """
    
    @staticmethod
    def plot_normalized_trace(t_plot, mean_norm, output_filename="flip_D_over_D0_poster.png"):
        """
        Create and save a plot of the normalized trace.
        """
        plt.figure(figsize=(4, 3), dpi=300)
        plt.plot(t_plot, mean_norm, "#000000", linewidth=1.5)
        
        plt.xlabel("Time [s]", fontsize=10)
        plt.ylabel(r"Normalised position $D/D_0$", fontsize=10)
        
        plt.ylim(-0.1, 1.1)
        plt.xlim(0, t_plot[-1])
        
        ax = plt.gca()
        ax.tick_params(axis="both", which="both",
                       labelsize=9, width=1, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=600, bbox_inches="tight")
        plt.show()
        
        print(f"Saved poster figure to: {output_filename}")


def find_tracking_files(pattern):
    """
    Find all tracking files matching the pattern.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"script_dir: {script_dir}")
    print(f"cwd: {os.getcwd()}")
    
    files = glob.glob(os.path.join(script_dir, pattern))
    print(f"glob pattern: {os.path.join(script_dir, pattern)}")
    print(f"found files: {files}")
    
    return files


def process_all_files(files, config):
    """
    Process all tracking files and extract segments.
    """
    processor = FlipDataProcessor(config)
    segments = []
    dts = []
    min_duration = None
    
    for f in files:
        t_seg, x_seg, dt = processor.process_file(f)
        if t_seg is None:
            continue
        
        dts.append(dt)
        dur_here = t_seg[-1] - t_seg[0]
        if (min_duration is None) or (dur_here < min_duration):
            min_duration = dur_here
        
        segments.append((t_seg, x_seg))
    
    if not segments:
        raise RuntimeError("No usable flips found")
    
    return segments, dts, min_duration


def create_common_time_axis(dts, min_duration, pre_time):
    """
    Create a common time axis for all segments.
    """
    common_dt = np.median(dts)
    t_start = -pre_time
    t_end = min_duration - pre_time
    return np.arange(t_start, t_end, common_dt)


def normalize_all_segments(segments, t_common, t_end, config):
    """
    Normalize all segments and interpolate to common time axis.
    """
    normalizer = FlipNormalizer(config)
    all_norm = []
    
    for (t_seg, x_seg) in segments:
        x_start, x_final, x_norm = normalizer.normalize_segment(t_seg, x_seg, t_end)
        if x_norm is None:
            continue
        
        x_interp = np.interp(t_common, t_seg, x_norm)
        all_norm.append(x_interp)
    
    all_norm = np.vstack(all_norm)
    return all_norm.mean(axis=0)


def save_results(t_plot, mean_norm, output_csv):
    """
    Save the combined flip trace to CSV.
    """
    out_df = pd.DataFrame({
        "t_s": t_plot,
        "D_over_D0": mean_norm
    })
    out_df.to_csv(output_csv, index=False)
    print(f"Saved combined flip trace to: {output_csv}")


def main():
    """
    Main function to run the flip analysis.
    """
    # Initialize configuration
    config = FlipAnalysisConfig()
    
    # Find all tracking files
    files = find_tracking_files(config.pattern)
    
    # Process all files
    segments, dts, min_duration = process_all_files(files, config)
    
    # Create common time axis
    t_common = create_common_time_axis(dts, min_duration, config.pre_time)
    t_end = min_duration - config.pre_time
    
    # Normalize all segments
    mean_norm = normalize_all_segments(segments, t_common, t_end, config)
    
    # Shift time axis so it starts at 0
    t_plot = t_common + config.pre_time
    
    # Create visualization
    visualizer = FlipVisualizer()
    visualizer.plot_normalized_trace(t_plot, mean_norm)
    
    # Save results
    save_results(t_plot, mean_norm, config.output_csv)


if __name__ == "__main__":
    main()
