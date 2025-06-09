#!/usr/bin/env python3
"""
Create architecture diagram for ModelSEEDagent
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

# Color scheme
colors = {
    "interface": "#E3F2FD",  # Light blue
    "agent": "#BBDEFB",  # Medium blue
    "tools": "#90CAF9",  # Blue
    "subsystem": "#64B5F6",  # Darker blue
    "llm": "#FFE082",  # Yellow
    "storage": "#C8E6C9",  # Light green
    "border": "#1976D2",  # Dark blue
}


def draw_box(ax, x, y, width, height, text, color, text_size=10, bold=False):
    """Draw a rounded rectangle with text"""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=colors["border"],
        linewidth=1.5,
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=text_size,
        weight=weight,
        wrap=True,
    )


def draw_arrow(ax, start_x, start_y, end_x, end_y, style="->", color="black"):
    """Draw an arrow between two points"""
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle=style, color=color, lw=1.5),
    )


# Title
ax.text(
    8,
    11.5,
    "ModelSEEDagent Architecture",
    horizontalalignment="center",
    fontsize=18,
    weight="bold",
)

# === USER INTERFACES LAYER ===
draw_box(
    ax,
    0.5,
    9.5,
    3.5,
    1.2,
    "Interactive CLI\n(Conversational AI)",
    colors["interface"],
    11,
    True,
)
draw_box(
    ax,
    4.5,
    9.5,
    3.5,
    1.2,
    "Standard CLI\n(Command-based)",
    colors["interface"],
    11,
    True,
)
draw_box(
    ax, 8.5, 9.5, 3.5, 1.2, "Python API\n(Direct access)", colors["interface"], 11, True
)
draw_box(
    ax, 12.5, 9.5, 3, 1.2, "Web Interface\n(Future)", colors["interface"], 11, True
)

# === AGENT LAYER ===
# RealTime Agent
draw_box(
    ax,
    0.5,
    7.5,
    7,
    1.5,
    "RealTimeMetabolicAgent\n• Dynamic AI decision-making\n• Custom workflow engine\n• Phase 8 capabilities",
    colors["agent"],
    10,
    True,
)

# LangGraph Agent
draw_box(
    ax,
    8.5,
    7.5,
    7,
    1.5,
    "LangGraphMetabolicAgent\n• StateGraph workflows\n• Parallel execution\n• State persistence & checkpointing",
    colors["agent"],
    10,
    True,
)

# === PHASE 8 SUBSYSTEMS (for RealTime Agent) ===
draw_box(
    ax,
    0.5,
    5.8,
    3.2,
    1,
    "Reasoning Chains\n• Multi-step planning\n• Dynamic adaptation",
    colors["subsystem"],
    9,
)
draw_box(
    ax,
    4.2,
    5.8,
    3.2,
    1,
    "Hypothesis System\n• Hypothesis generation\n• Testing workflows",
    colors["subsystem"],
    9,
)
draw_box(
    ax,
    0.5,
    4.5,
    3.2,
    1,
    "Collaborative Reasoning\n• User interaction\n• Uncertainty detection",
    colors["subsystem"],
    9,
)
draw_box(
    ax,
    4.2,
    4.5,
    3.2,
    1,
    "Pattern Memory\n• Cross-model learning\n• Experience-based optimization",
    colors["subsystem"],
    9,
)

# === TOOL INTEGRATION ===
draw_box(
    ax,
    8.5,
    5.8,
    7,
    1,
    "Enhanced Tool Integration\n• Intelligent tool selection • Performance monitoring • Workflow visualization",
    colors["subsystem"],
    9,
)

# === TOOL LAYER ===
# COBRA Tools
draw_box(
    ax,
    0.5,
    2.8,
    4.5,
    1.2,
    "COBRA.py Tools\n• FBA, FVA, Gene deletion\n• Minimal media, Auxotrophy\n• Essentiality analysis",
    colors["tools"],
    9,
    True,
)

# ModelSEED Tools
draw_box(
    ax,
    5.5,
    2.8,
    4.5,
    1.2,
    "ModelSEED Tools\n• Model building, Gapfilling\n• RAST annotation\n• Compatibility testing",
    colors["tools"],
    9,
    True,
)

# Biochemistry Tools
draw_box(
    ax,
    10.5,
    2.8,
    4.5,
    1.2,
    "Biochemistry Tools\n• Entity resolution\n• Database search\n• Compound/reaction lookup",
    colors["tools"],
    9,
    True,
)

# === LLM LAYER ===
draw_box(ax, 2, 1, 3, 1, "Argo Gateway\n(ANL)", colors["llm"], 10, True)
draw_box(ax, 6, 1, 3, 1, "OpenAI API\n(GPT models)", colors["llm"], 10, True)
draw_box(ax, 10, 1, 3, 1, "Local Models\n(Ollama)", colors["llm"], 10, True)

# === STORAGE LAYER ===
draw_box(ax, 0.5, 0.2, 2.5, 0.6, "Audit Logs", colors["storage"], 9)
draw_box(ax, 3.5, 0.2, 2.5, 0.6, "Session Data", colors["storage"], 9)
draw_box(ax, 6.5, 0.2, 2.5, 0.6, "Learning Memory", colors["storage"], 9)
draw_box(ax, 9.5, 0.2, 2.5, 0.6, "Model Cache", colors["storage"], 9)
draw_box(ax, 12.5, 0.2, 2.5, 0.6, "Visualizations", colors["storage"], 9)

# === ARROWS ===
# Interface to Agents
draw_arrow(ax, 2.25, 9.5, 4, 9)  # Interactive CLI -> RealTime
draw_arrow(ax, 6.25, 9.5, 12, 9)  # Standard CLI -> LangGraph
draw_arrow(ax, 10.25, 9.5, 12, 9)  # Python API -> LangGraph

# Agents to Subsystems
draw_arrow(ax, 2, 7.5, 2, 6.8)  # RealTime -> Reasoning
draw_arrow(ax, 4, 7.5, 5.8, 6.8)  # RealTime -> Hypothesis
draw_arrow(ax, 3, 7.5, 2, 5.5)  # RealTime -> Collaborative
draw_arrow(ax, 5, 7.5, 5.8, 5.5)  # RealTime -> Pattern Memory

draw_arrow(ax, 12, 7.5, 12, 6.8)  # LangGraph -> Tool Integration

# To Tools
draw_arrow(ax, 4, 4.5, 2.75, 4)  # Subsystems -> COBRA
draw_arrow(ax, 4, 4.5, 7.75, 4)  # Subsystems -> ModelSEED
draw_arrow(ax, 12, 5.8, 12.75, 4)  # Tool Integration -> Biochem

# To LLMs
draw_arrow(ax, 4, 7.5, 3.5, 2)  # RealTime -> Argo
draw_arrow(ax, 12, 7.5, 7.5, 2)  # LangGraph -> OpenAI
draw_arrow(ax, 8, 7.5, 11.5, 2)  # Both -> Local

# Legend
legend_elements = [
    mpatches.Rectangle(
        (0, 0), 1, 1, facecolor=colors["interface"], label="User Interfaces"
    ),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors["agent"], label="AI Agents"),
    mpatches.Rectangle(
        (0, 0), 1, 1, facecolor=colors["subsystem"], label="Agent Subsystems"
    ),
    mpatches.Rectangle(
        (0, 0), 1, 1, facecolor=colors["tools"], label="Tool Categories"
    ),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors["llm"], label="LLM Backends"),
    mpatches.Rectangle(
        (0, 0), 1, 1, facecolor=colors["storage"], label="Storage & Persistence"
    ),
]

ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig(
    "/Users/jplfaria/repos/ModelSEEDagent/architecture_diagram.png",
    dpi=300,
    bbox_inches="tight",
)
print("Architecture diagram saved as 'architecture_diagram.png'")
