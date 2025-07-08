import plotly.graph_objs as go

def plot_stage_chart(stage_counts):
    total = sum(stage_counts) or 1
    percents = [round(c / total * 100, 2) for c in stage_counts]
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    colors = ['#BDC3C7', '#95A5A6', '#7F8C8D']

    fig = go.Figure()
    for stage, percent, color in zip(stages, percents, colors):
        fig.add_trace(go.Scatter(x=[stage], y=[percent], mode='markers', marker=dict(size=12, color=color), showlegend=False))

    fig.add_trace(go.Scatter(
        x=stages, y=percents, mode='lines+text',
        text=[f"{p}%" for p in percents], textposition="top center",
        line=dict(color='red', width=3)
    ))
    fig.update_layout(title="Tumor Stage % Distribution", xaxis_title="Stage", yaxis_title="Percentage", showlegend=False)
    return fig

def plot_bar_chart(stage_counts):
    fig = go.Figure([go.Bar(x=["Stage 1", "Stage 2", "Stage 3"], y=stage_counts,
                            marker_color=['#BDC3C7', '#95A5A6', '#7F8C8D'])])
    fig.update_layout(title="Tumor Stage Raw Count", xaxis_title="Stage", yaxis_title="Count")
    return fig
