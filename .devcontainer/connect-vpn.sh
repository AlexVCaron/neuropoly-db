#!/usr/bin/env bash
# connect-vpn.sh — Interactive PolyQuartz VPN connector.
#
# Calls poly_vpn.py to obtain a webvpn cookie via the Playwright/Okta flow,
# then hands it to openconnect.
#
# ── Network-mode host warning ────────────────────────────────────────────────
# This devcontainer runs with network_mode: host.  openconnect creates a tun
# interface that is visible on the DOCKER HOST's network stack, not just inside
# the container.  On a shared Linux server (rosenberg, joplin, etc.) this will
# alter routing for ALL users on that machine.  Do not use on shared servers
# without coordinating with other users first.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ANSI helpers
_RESET="\033[0m"
_BOLD="\033[1m"
_RED="\033[31m"
_GREEN="\033[32m"
_YELLOW="\033[33m"
_AMBER="\033[38;5;214m"

_ok()    { echo -e "  ${GREEN}✓${_RESET} ${_BOLD}${1}${_RESET}"; }
_warn()  { echo -e "  ${_AMBER}⚠${_RESET}  ${1}"; }
_error() { echo -e "  ${_RED}✗${_RESET} ${_BOLD}${1}${_RESET}" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/workspaces/neuropoly-db/.venv/bin/python"

VPN_PID_FILE="/tmp/vpn.pid"
VPN_USER_FILE="/tmp/.vpn_connected_user"

# ── Guard: already connected? ────────────────────────────────────────────────
if [[ -f "$VPN_PID_FILE" ]] && [[ -d "/proc/$(cat "$VPN_PID_FILE" 2>/dev/null)" ]]; then
    _vpn_user=$(cat "$VPN_USER_FILE" 2>/dev/null || echo "unknown")
    _ok "VPN is already connected as ${_vpn_user}."
    _warn "Run \`vpndisconnect\` to disconnect."
    exit 0
fi
# Clean up stale state files in case openconnect died without cleanup
rm -f "$VPN_PID_FILE" "$VPN_USER_FILE"

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${_BOLD}${_YELLOW}╔══════════════════════════════════════════════════════╗${_RESET}"
echo -e "  ${_BOLD}${_YELLOW}║          Connecting to PolyMTL VPN (PolyQuartz)      ║${_RESET}"
echo -e "  ${_BOLD}${_YELLOW}╠══════════════════════════════════════════════════════╣${_RESET}"
echo -e "  ${_BOLD}${_YELLOW}║  A headless browser will navigate the Okta portal.   ║${_RESET}"
echo -e "  ${_BOLD}${_YELLOW}║  You will be prompted for your CAS credentials,      ║${_RESET}"
echo -e "  ${_BOLD}${_YELLOW}║  then asked to approve a push notification.          ║${_RESET}"
echo -e "  ${_BOLD}${_YELLOW}╚══════════════════════════════════════════════════════╝${_RESET}"
echo ""

# ── Sanity checks ────────────────────────────────────────────────────────────
if [[ ! -x "$VENV_PYTHON" ]]; then
    _error "Virtual-env Python not found at $VENV_PYTHON"
    _error "Run the postCreate setup first, then try again."
    exit 1
fi

if ! command -v openconnect > /dev/null 2>&1; then
    _error "openconnect is not installed. Reinstall the devcontainer."
    exit 1
fi

# ── Fetch cookie (all interactive output goes to stderr via poly_vpn.py) ─────
VPN_COOKIE="$("$VENV_PYTHON" "$SCRIPT_DIR/poly_vpn.py")"

if [[ -z "$VPN_COOKIE" ]]; then
    _error "No VPN cookie was returned. Aborting."
    exit 1
fi

# ── Connect (background) ─────────────────────────────────────────────────────
echo ""
echo -e "  ${_BOLD}Handing cookie to openconnect…${_RESET}"
echo ""
echo "$VPN_COOKIE" | sudo openconnect \
    --protocol=anyconnect \
    --authgroup=PolyQuartz \
    --cookie-on-stdin \
    --reconnect-timeout 20 \
    --background \
    --pid-file="$VPN_PID_FILE" \
    --syslog \
    https://ssl.vpn.polymtl.ca/

# Give openconnect a moment to bring up the tun interface.
sleep 3
_vpn_user=$(cat "$VPN_USER_FILE" 2>/dev/null || echo "unknown")
if ip link show 2>/dev/null | grep -qE '^[0-9]+: tun'; then
    echo ""
    echo -e "  ${_BOLD}${_GREEN}╔══════════════════════════════════════════════════════╗${_RESET}"
    echo -e "  ${_BOLD}${_GREEN}║  ✓  Connected to PolyMTL VPN as ${_vpn_user}$(printf '%*s' $((21 - ${#_vpn_user})) '')║${_RESET}"
    echo -e "  ${_BOLD}${_GREEN}║     Run: vpndisconnect                               ║${_RESET}"
    echo -e "  ${_BOLD}${_GREEN}╚══════════════════════════════════════════════════════╝${_RESET}"
    echo ""
else
    _error "openconnect started but no tun interface appeared. Check: journalctl -t openconnect"
    rm -f "$VPN_PID_FILE" "$VPN_USER_FILE"
    exit 1
fi
