# CELL 3.1: ANIMATED SWEEPING CORRELATION TIMELINE ENGINE

from Dependencies import *      # Import specific functions required from Dependencies
from AI_unsupervised import *   # Import specific functions required from AI_unsupervised

print("--- Running Animated Sweeping Correlation Timeline Engine ---")

# 1. Compute dynamic lookback window
rolling_window = 14
clean_df["Asset_Correlation"] = (
    clean_df["BNB_Returns"].rolling(window=rolling_window).corr(clean_df["BTC_Returns"])
)
clean_df = clean_df.dropna()

# Explicitly pass numeric_only=True to prevent text column aggregation failures
animated_df = clean_df.resample("W").mean(numeric_only=True).dropna()
total_steps = len(animated_df)

regime_names = {
    0: "Regime 0: Balanced Stability",
    1: "Regime 1: Market Indecision/Chop",
    2: "Regime 2: High-Volatility Shift",
}
# Define matching signature hex codes
regime_colors = {0: "#2CA02C", 1: "#FF7F0E", 2: "#D62728"}

# 2. Build the Initial Base Frame (The starting view at Week 1)
initial_idx = animated_df.index[:15]  # Show a small initial window
fig = go.Figure()

# Background trace tracking the historical correlation curve line
fig.add_trace(
    go.Scatter(
        x=animated_df.index,
        y=animated_df["Asset_Correlation"],
        mode="lines",
        line=dict(width=1.5, color="#E0E0E0"),
        name="Full Historical Track",
        hoverinfo="skip",
    )
)

# Animated glowing trendline that sweeps across the screen
fig.add_trace(
    go.Scatter(
        x=initial_idx,
        y=animated_df["Asset_Correlation"].iloc[:15],
        mode="lines+markers",
        line=dict(width=3.5, color="#1F4E79"),
        marker=dict(size=5, color="#F7931A"),
        name="Dynamic Correlation Window",
    )
)

# 3. Construct the Sequential Sweeping Animation Frames
frames = []
for i in range(15, total_steps):
    current_date = animated_df.index[i]
    current_regime = int(animated_df["Market_Regime_Cluster"].iloc[i])
    current_corr = animated_df["Asset_Correlation"].iloc[i]

    # Cumulative data subset up to the current sweep position
    sweep_idx = animated_df.index[:i]
    sweep_data = animated_df["Asset_Correlation"].iloc[:i]

    frames.append(
        go.Frame(
            data=[
                go.Scatter(x=animated_df.index, y=animated_df["Asset_Correlation"]),
                go.Scatter(x=sweep_idx, y=sweep_data),
            ],
            name=str(current_date.strftime("%Y-%m-%d")),
            layout=dict(
                title=dict(
                    text=f"<b>Animated BNB-BTC Correlation Sweep</b><br>"
                    f"Current Date: {current_date.strftime('%Y-%m-%d')} | "
                    f"Correlation: {current_corr:.2f} | <b><span style='color:{regime_colors[current_regime]}'>{regime_names[current_regime]}</span></b>"
                )
            ),
        )
    )

fig.frames = frames

# 4. Interface Configurations & Playback Layout Architecture
fig.update_layout(
    title=dict(
        text="<b>Animated BNB-BTC Correlation Sweep</b><br>Click Play to begin the timeline tracking run...",
        font=dict(size=15, color="#1A1A1A"),
        x=0.05,
    ),
    xaxis=dict(title="Timeline History", gridcolor="#E5E5E5", showgrid=True),
    yaxis=dict(
        title="Pearson Correlation (r)",
        range=[-1.05, 1.05],
        gridcolor="#E5E5E5",
        showgrid=True,
    ),
    margin=dict(l=50, r=30, b=80, t=90),
    plot_bgcolor="white",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            x=0.05,
            y=-0.15,
            xanchor="left",
            yanchor="top",
            direction="left",
            buttons=[
                dict(
                    label="▶ Play Sweeper",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=50, redraw=False),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=False),
                            mode="immediate",
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )
    ],
)

# EXPORT & PRESENT WORKSPACE
output_html_path = "correlation_sweeper.html"
fig.write_html(output_html_path)
print(f"[SUCCESS] Dynamic Timeline Sweeper saved to: '{output_html_path}'")
print("[RENDER] Rendering interactive line sweep workspace below.")
fig.show()