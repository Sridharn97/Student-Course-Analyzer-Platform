"""UI Utilities for the Student Course Analyzer Platform."""

import plotly.graph_objects as go

def apply_premium_plotly_layout(fig):
    """
    Applies a standardized, premium, and clean layout to a Plotly figure.
    This matches the custom CSS and global Streamlit theme.
    """
    # Define color palette matching our indigo theme
    colorway = [
        '#4F46E5', # Indigo
        '#06B6D4', # Cyan
        '#10B981', # Emerald
        '#F59E0B', # Amber
        '#EC4899', # Pink
        '#8B5CF6', # Violet
    ]
    
    fig.update_layout(
        font_family="Inter, Roboto, sans-serif",
        font_color="#334155",
        title_font_color="#0F172A",
        title_font_size=18,
        colorway=colorway,
        
        # Transparent backgrounds so it blends seamlessly with Streamlit cards
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        
        # Cleaner hover labels
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, Roboto, sans-serif"
        ),
        
        # Clean margins
        margin=dict(t=50, l=10, r=10, b=10),
    )
    
    # Clean up axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E2E8F0",
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        tickfont_color="#64748B"
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E2E8F0",
        zeroline=False,
        showline=False,
        tickfont_color="#64748B"
    )
    
    return fig
