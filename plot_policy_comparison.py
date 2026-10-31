import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    df = pd.read_csv('output/results/multihub/route_baseline_summary.csv')

    policies_to_plot = ["Historical", "Maximum-EE", "Multi-objective"]
    df_plot = df[df['strategy'].isin(policies_to_plot)].copy()

    df_plot['strategy'] = pd.Categorical(df_plot['strategy'], categories=policies_to_plot, ordered=True)
    df_plot = df_plot.sort_values('strategy')

    labels = ["Historical (P0)", "Max-EE (P3)", "Multi-objective (P4)"]
    ee_means = df_plot['mean_predicted_ee'].values
    energy_means = df_plot['mean_predicted_energy'].values

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Route-Selection Policy', fontsize=12)
    ax1.set_ylabel('Mean Predicted EE (m/Wh)', color=color1, fontsize=12)
    rects1 = ax1.bar(x - width/2, ee_means, width, label='EE (m/Wh)', color=color1, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Adjust y-axis limits to make differences visible
    ax1.set_ylim(min(ee_means) * 0.95, max(ee_means) * 1.05)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Mean Predicted Energy (Wh)', color=color2, fontsize=12)
    rects2 = ax2.bar(x + width/2, energy_means, width, label='Energy (Wh)', color=color2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Adjust y-axis limits to make differences visible
    ax2.set_ylim(min(energy_means) * 0.95, max(energy_means) * 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    
    plt.title('Comparison of Historical and Candidate Policies\nP3 maximizes EE, P4 balances EE and Energy', fontsize=13)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, ax1)
    autolabel(rects2, ax2)

    fig.tight_layout()
    os.makedirs('output/figures/multihub', exist_ok=True)
    out_path = 'output/figures/multihub/policy_comparison_p3_p4.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
