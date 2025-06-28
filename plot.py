import csv
import matplotlib.pyplot as plt

# Load CSV data
csv_path = 'LightGlue/matching_results.csv'
pair_indices = []
times_ms = []
matches = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pair_indices.append(int(row['PairIndex']))
        time_sec = float(row['Time(s)'])
        times_ms.append(time_sec * 1000)  # convert seconds to milliseconds
        matches.append(int(row['Matches']))

# Find peak values
max_time = max(times_ms)
max_time_index = pair_indices[times_ms.index(max_time)]

max_matches = max(matches)
max_matches_index = pair_indices[matches.index(max_matches)]

# Calculate statistics
avg_time = sum(times_ms) / len(times_ms)
avg_matches = sum(matches) / len(matches)

# Create Figure 1: Time Performance
plt.figure(figsize=(12, 8))
plt.title('⏱️ LightGlue Processing Time per Image Pair', fontsize=16, pad=20, fontweight='bold')

color_time = 'tab:orange'
plt.xlabel('Image Pair Index', fontsize=12)
plt.ylabel('Processing Time (ms)', fontsize=12, color=color_time)
plt.plot(pair_indices, times_ms, color=color_time, marker='o', linewidth=2, markersize=6, label='Processing Time')
plt.fill_between(pair_indices, times_ms, color=color_time, alpha=0.2)

# Add horizontal line for average
plt.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_time:.1f}ms')

# Annotate max time
plt.annotate(f'Peak Time: {max_time:.1f}ms\n(Pair {max_time_index})',
             xy=(max_time_index, max_time),
             xytext=(max_time_index + len(pair_indices)*0.1, max_time + max_time*0.1),
             arrowprops=dict(arrowstyle='->', color=color_time, lw=1.5),
             fontsize=11, color=color_time, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color_time, alpha=0.8))

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()

# Save first plot
plt.savefig('dataset/match_vis/time_performance_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Create Figure 2: Matches Performance
plt.figure(figsize=(12, 8))
plt.title('🎯 LightGlue Feature Matches per Image Pair', fontsize=16, pad=20, fontweight='bold')

color_matches = 'tab:green'
plt.xlabel('Image Pair Index', fontsize=12)
plt.ylabel('Number of Feature Matches', fontsize=12, color=color_matches)
plt.plot(pair_indices, matches, color=color_matches, marker='x', linewidth=2, markersize=8, label='Feature Matches')
plt.fill_between(pair_indices, matches, color=color_matches, alpha=0.2)

# Add horizontal line for average
plt.axhline(y=avg_matches, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_matches:.1f} matches')

# Annotate max matches
plt.annotate(f'Peak Matches: {max_matches}\n(Pair {max_matches_index})',
             xy=(max_matches_index, max_matches),
             xytext=(max_matches_index + len(pair_indices)*0.1, max_matches + max_matches*0.1),
             arrowprops=dict(arrowstyle='->', color=color_matches, lw=1.5),
             fontsize=11, color=color_matches, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color_matches, alpha=0.8))

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right')
plt.tight_layout()

# Save second plot
plt.savefig('dataset/match_vis/matches_performance_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("="*50)
print("📊 LIGHTGLUE PERFORMANCE SUMMARY")
print("="*50)
print(f"Total Image Pairs Processed: {len(pair_indices)}")
print(f"Time Statistics:")
print(f"  • Average Processing Time: {avg_time:.2f} ms")
print(f"  • Maximum Processing Time: {max_time:.2f} ms (Pair {max_time_index})")
print(f"  • Minimum Processing Time: {min(times_ms):.2f} ms (Pair {pair_indices[times_ms.index(min(times_ms))]})")
print(f"Match Statistics:")
print(f"  • Average Matches: {avg_matches:.1f}")
print(f"  • Maximum Matches: {max_matches} (Pair {max_matches_index})")
print(f"  • Minimum Matches: {min(matches)} (Pair {pair_indices[matches.index(min(matches))]})")
print("="*50)