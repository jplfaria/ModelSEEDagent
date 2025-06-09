"""
Argo Gateway Health Check for ModelSEEDagent Interactive CLI

Checks Argo Gateway model availability and displays current configuration
when starting the interactive CLI.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_random_exponential

console = Console()

# --------------------------------- ARGO HEALTH-CHECK ---------------------------------

# ❶ User config
ARGO_USER = os.environ.get("ARGO_USER", "jplfaria")

# ❷ Auth header (comment out if the gateway is IP-whitelisted)
AUTH_HEADERS = {
    # "Authorization": f"Bearer {os.environ['ARGO_TOKEN']}",
    "Content-Type": "application/json",
}

# ❸ Request timeout (seconds)
TIMEOUT = 10  # Shorter timeout for CLI startup

# Model lists
CHAT_MODELS_ALL: Dict[str, List[str]] = {
    "prod": [
        "gpt35",
        "gpt35large",
        "gpt4",
        "gpt4large",
        "gpt4turbo",
        "gpt4o",
        "gpto1preview",
    ],
    "test": [],
    "dev": [
        "gpto1preview",
        "gpto1mini",
        "gpto3mini",
        "gpto1",
        "gpt4o",
        "gpt4olatest",
    ],
}

EMBED_MODELS_ALL = ["ada002", "v3large", "v3small"]

BASES = {
    "prod": "https://apps.inside.anl.gov/argoapi/api/v1/resource",
    "test": "https://apps-test.inside.anl.gov/argoapi/api/v1/resource",
    "dev": "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource",
}


def _chat_payload(model: str, prompt: str = "ping") -> Dict:
    """Build minimal payload for each model family."""
    if model.startswith(("gpto1", "gpto3")):
        return {
            "user": ARGO_USER,
            "model": model,
            "prompt": [prompt],
            "max_completion_tokens": 8,
        }
    else:
        return {
            "user": ARGO_USER,
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 0,
            "max_tokens": 1,
        }


@retry(wait=wait_random_exponential(min=1, max=4), stop=stop_after_attempt(2))
def _post_json(url: str, payload: Dict) -> Tuple[bool, str]:
    """POST JSON with quick retries for CLI startup."""
    try:
        r = requests.post(url, headers=AUTH_HEADERS, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return True, "✅"
    except requests.exceptions.RequestException as exc:
        return False, f"❌ {type(exc).__name__}"


def check_argo_status_quick() -> Dict[str, Dict[str, str]]:
    """Quick Argo status check for CLI startup."""
    results = {}

    # Check a few key models only for speed
    key_models = {
        "prod": ["gpt4o", "gpto1preview"],
        "dev": ["gpt4o", "gpto1", "gpto1mini"],
    }

    for env, models in key_models.items():
        results[env] = {}
        chat_url = f"{BASES[env]}/chat/"

        for model in models:
            try:
                ok, msg = _post_json(chat_url, _chat_payload(model))
                results[env][model] = msg
            except Exception:
                results[env][model] = "❌ Error"

    return results


def get_current_config() -> Dict[str, str]:
    """Get current ModelSEEDagent configuration."""
    config = {}

    # Check environment variables first
    config["ARGO_USER"] = os.environ.get("ARGO_USER", "Not set")

    # Check for the CLI config file (created by modelseed-agent setup)
    cli_config_file = os.path.expanduser("~/.modelseed-agent-cli.json")
    if os.path.exists(cli_config_file):
        try:
            with open(cli_config_file, "r") as f:
                cli_config = json.load(f)
                # Extract relevant config from CLI config
                # Backend is saved at top level
                if "llm_backend" in cli_config:
                    config["backend"] = cli_config["llm_backend"]
                # Model name is in llm_config
                if (
                    "llm_config" in cli_config
                    and "model_name" in cli_config["llm_config"]
                ):
                    config["model_name"] = cli_config["llm_config"]["model_name"]
                # Check for user field in llm_config (this is where setup saves it)
                if "llm_config" in cli_config and "user" in cli_config["llm_config"]:
                    config["ARGO_USER"] = cli_config["llm_config"]["user"]
                # Legacy check for argo_user
                elif "llm" in cli_config and "argo_user" in cli_config["llm"]:
                    config["ARGO_USER"] = cli_config["llm"]["argo_user"]
                # If ARGO_USER not in llm config, check global config
                if config["ARGO_USER"] == "Not set" and "argo_user" in cli_config:
                    config["ARGO_USER"] = cli_config["argo_user"]
                config["config_file"] = "~/.modelseed-agent-cli.json"
        except Exception as e:
            config["config_file"] = f"Error reading config: {e}"
    else:
        config["config_file"] = "No config file found (run 'modelseed-agent setup')"

    # Try fallback config files
    if config.get("config_file", "").startswith("No"):
        fallback_config_file = os.path.expanduser("~/.modelseed_config.json")
        if os.path.exists(fallback_config_file):
            try:
                with open(fallback_config_file, "r") as f:
                    file_config = json.load(f)
                    config.update(file_config)
                    config["config_file"] = "~/.modelseed_config.json (fallback)"
            except Exception:
                pass

    return config


def display_argo_health():
    """Display Argo health check and configuration for CLI startup."""

    console.print("\n🔍 [bold cyan]Argo Gateway Health Check[/bold cyan]")

    # Quick status check
    with console.status("🔄 Checking Argo Gateway...", spinner="dots"):
        argo_status = check_argo_status_quick()

    # Display results in a compact table
    if argo_status:
        status_table = Table(title="🌐 Argo Model Status", box=box.SIMPLE_HEAVY)
        status_table.add_column("Environment", style="bold")
        status_table.add_column("Model", style="bold")
        status_table.add_column("Status", justify="center")

        for env, models in argo_status.items():
            for model, status in models.items():
                status_table.add_row(env.upper(), model, status)

        console.print(status_table)
    else:
        console.print("❌ [red]Could not connect to Argo Gateway[/red]")

    # Current configuration
    config = get_current_config()

    config_content = []

    # Show ARGO_USER
    argo_user = config.get("ARGO_USER", "Not set")
    config_content.append(f"🔑 **ARGO_USER:** {argo_user}")

    # Show backend and model
    backend = config.get("backend", "Not configured")
    model_name = config.get("model_name", "Not configured")

    config_content.append(f"🔧 **Backend:** {backend}")
    config_content.append(f"🤖 **Model:** {model_name}")

    # Show config file status
    config_file_status = config.get("config_file", "No config file")
    config_content.append(f"📁 **Config File:** {config_file_status}")

    config_panel = Panel(
        "\n".join(config_content),
        title="[bold green]🛠️ Current Configuration[/bold green]",
        border_style="green",
    )
    console.print(config_panel)

    # Usage tips
    tips = [
        "💡 **Tip:** Use `modelseed-agent setup --backend argo --model gpto1` to configure",
        "💡 **Tip:** Set ARGO_USER environment variable for your username",
        "💡 **Tip:** o1 models (gpto1, gpto1mini) are slower but more capable",
    ]

    console.print("\n" + "\n".join(tips))
    console.print()


def check_model_availability(model_name: str, env: str = "dev") -> bool:
    """Check if a specific model is available."""
    if env not in BASES:
        return False

    chat_url = f"{BASES[env]}/chat/"
    try:
        ok, _ = _post_json(chat_url, _chat_payload(model_name))
        return ok
    except Exception:
        return False


if __name__ == "__main__":
    display_argo_health()
