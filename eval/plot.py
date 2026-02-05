import matplotlib.pyplot as plt
import numpy as np

PURPLE = "#741b47"
ORANGE = "#ff9100"
GRAY = "#666666"

w = 1
mean_zlib_compression_rates_path = f"./metrics/mean_zlib_compression_rates_seed=108_l=1025_w={w}_nc=10_ns=50_t=0.1_v=26.npy"
mean_log_loss_bits_path = f"./metrics/mean_log_loss_bits_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy"

# metrics_dir = "../eval"

# Set fontsize and style
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.family': 'DejaVu Sans'})

compression_rates_zlib = np.load(mean_zlib_compression_rates_path)
compression_rates_transformer = np.load(mean_log_loss_bits_path)

# Plot the middle 256 points
zoom_len = 128
mid = len(compression_rates_zlib) // 2
start = mid - zoom_len // 2
end = mid + zoom_len // 2


# # Plot the entire range
# start = 0
# end = len(compression_rates_zlib)

zlib_zoom = compression_rates_zlib[start:end]
transformer_zoom = compression_rates_transformer[start:end]

# Save the zoomed compression rates
np.save(f"metrics/zlib_zoom_len={zoom_len}_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy", zlib_zoom)
np.save(f"metrics/transformer_zoom_len={zoom_len}_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy", transformer_zoom)

plt.plot(range(start, end), zlib_zoom, label="zlib", color=GRAY)
plt.plot(range(start, end), transformer_zoom, label="transformer", color=PURPLE)
plt.xlabel("sequence length")
plt.ylabel("compression rate (bpc)")
plt.title(f"window length: {w}")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/plot_delta_zlib_vs_transformer_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.png")