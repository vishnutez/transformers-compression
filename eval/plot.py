import matplotlib.pyplot as plt
import numpy as np

PURPLE = "#741b47"
ORANGE = "#ff9100"
GRAY = "#666666"

w = 1
mean_optimal_loss_bits = np.load(f"metrics/mean_optimal_log_loss_bits_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy")
mean_zlib_compression_rates = np.load(f"metrics/mean_zlib_compression_rates_seed=108_l=1025_w={w}_nc=10_ns=50_t=0.1_v=26.npy")
mean_log_loss_bits = np.load(f"metrics/mean_log_loss_bits_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy")

# Set fontsize and style
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'DejaVu Sans'})


mid = len(mean_zlib_compression_rates) // 2


# # Plot the middle 256 points
# zoom_len = 128
# start = mid - zoom_len // 2
# end = mid + zoom_len // 2


# Plot the entire range
start = 0
end = len(mean_zlib_compression_rates)
zoom_len = end - start

zlib_zoom = mean_zlib_compression_rates[start:end]
transformer_zoom = mean_log_loss_bits[start:end]
optimal_zoom = mean_optimal_loss_bits[start:end]

# # Save the zoomed compression rates
# np.save(f"metrics/zlib_zoom_len={zoom_len}_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy", zlib_zoom)
# np.save(f"metrics/transformer_zoom_len={zoom_len}_seed=108_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.npy", transformer_zoom)

avg_ent_rate_pre = np.load(f"metrics/avg_ent_rate_pre_t=0.1_v=26_nc=10.npy")
avg_ent_rate_post = np.load(f"metrics/avg_ent_rate_post_t=0.1_v=26_nc=10.npy")

# Construct the entropy rate curve by combining the pre and post entropy rate
entropy_rate = np.zeros(end - start)
entropy_rate[:mid] = avg_ent_rate_pre
entropy_rate[mid:] = avg_ent_rate_post

plt.plot(range(start, end), zlib_zoom, label="zlib", color=GRAY)
plt.plot(range(start, end), transformer_zoom, label="transformer", color=PURPLE)
plt.plot(range(start, end), optimal_zoom, label="optimal", color=ORANGE)
plt.plot(range(start, end), entropy_rate, label="entropy rate", color="k")
plt.xlabel("sequence length")
plt.ylabel("compression rate (bpc)")
plt.title(f"window length: {w}")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/full_zlib_vs_transformer_vs_optimal_l=1024_w={w}_nc=10_ns=50_t=0.1_v=26.png")