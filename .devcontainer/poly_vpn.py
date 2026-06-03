#!/usr/bin/env python3
"""
poly_vpn.py — Interactive PolyQuartz VPN cookie fetcher.

Uses Playwright + Chromium (headless) to complete the PolyQuartz / Okta
authentication flow for Polytechnique Montréal staff (pMatricule holders).

Output contract
---------------
- All progress / prompts / errors  → stderr  (ANSI-coloured, safe to print)
- The raw webvpn cookie value only → stdout  (no trailing newline, no colour)
  so the calling bash script can capture it cleanly with $().

Intended to be called by connect-vpn.sh, not directly by users.
"""

import getpass
import re
import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

# ---------------------------------------------------------------------------
# Target URL — PolyQuartz group logon page
# ---------------------------------------------------------------------------
POLYQUARTZ_URL = (
    "https://ssl.vpn.polymtl.ca/+CSCOE+/logon.html"
    "?reason=12&gmsg=4646594365627376797244686E65676D#form_title_text"
)

# Written on successful auth so connect-vpn.sh and terminal-header.sh can
# display the connected username without re-running any auth flow.
VPN_STATE_FILE = "/tmp/.vpn_connected_user"

# ---------------------------------------------------------------------------
# ANSI helpers — all output goes to stderr
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_AMBER = "\033[38;5;214m"  # 256-colour amber


def _err(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def _step(msg: str):
    _err(f"  {_CYAN}→{_RESET} {msg}")


def _ok(msg: str):
    _err(f"  {_GREEN}✓{_RESET} {_BOLD}{msg}{_RESET}")


def _warn(msg: str):
    _err(f"  {_AMBER}⚠{_RESET}  {msg}")


def _error(msg: str):
    _err(f"  {_RED}✗{_RESET} {_BOLD}{msg}{_RESET}")


def _prompt(msg: str) -> str:
    """Print a styled prompt to stderr and read a line from stdin (echoed)."""
    _err(f"  {_BOLD}{_WHITE}{msg}{_RESET} ", end="")
    sys.stderr.flush()
    return sys.stdin.readline().rstrip("\n")


def _prompt_password(msg: str) -> str:
    """Print a styled prompt to stderr and read a password (hidden)."""
    # getpass reads from /dev/tty so the password is never written to stdout/stderr
    return getpass.getpass(
        prompt=f"  {_BOLD}{_WHITE}{msg}{_RESET} ",
        stream=sys.stderr,
    )


def _banner():
    _err("")
    _err(
        f"  {_BOLD}{_YELLOW}╔══════════════════════════════════════════════════════╗{_RESET}"
    )
    _err(
        f"  {_BOLD}{_YELLOW}║     PolyMTL VPN — PolyQuartz interactive login       ║{_RESET}"
    )
    _err(
        f"  {_BOLD}{_YELLOW}╠══════════════════════════════════════════════════════╣{_RESET}"
    )
    _err(
        f"  {_BOLD}{_YELLOW}║  Step 1 — Enter your CAS username and password       ║{_RESET}"
    )
    _err(
        f"  {_BOLD}{_YELLOW}║  Step 2 — Approve the Okta push on your phone        ║{_RESET}"
    )
    _err(
        f"  {_BOLD}{_YELLOW}║  Step 3 — VPN cookie is extracted automatically      ║{_RESET}"
    )
    _err(
        f"  {_BOLD}{_YELLOW}╚══════════════════════════════════════════════════════╝{_RESET}"
    )
    _err("")


# ---------------------------------------------------------------------------
# Main Playwright flow
# ---------------------------------------------------------------------------
def fetch_vpn_cookie() -> str:
    """
    Drive the PolyQuartz / Okta login in a headless Chromium browser and
    return the raw `webvpn` cookie value.  Raises on failure.
    """
    with sync_playwright() as pw:
        # --no-sandbox is required inside Docker/devcontainers: without it,
        # Chromium creates its own network namespace and cannot reach external hosts.
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",  # Docker /dev/shm is 64 MB; use /tmp instead
                "--disable-gpu",
            ],
        )
        context = browser.new_context()
        page = context.new_page()

        try:
            # ── Step 1a: Load the VPN portal ──────────────────────────────
            _step("Loading PolyQuartz VPN portal…")
            page.goto(POLYQUARTZ_URL)
            page.wait_for_selector("#group_list", timeout=15_000)
            _step("VPN portal loaded — selecting PolyQuartz group…")

            # Click the Login button to initiate the SAML redirect
            page.locator("[name='Login']").click()

            # ── Step 1b: Wait for Okta / Polytechnique sign-in page ───────
            _step("Waiting for Polytechnique sign-in page…")
            page.wait_for_function(
                "document.title.includes('Polytechnique Montréal')",
                timeout=15_000,
            )
            _ok("Polytechnique sign-in page ready")

            # ── Step 2a: Username ─────────────────────────────────────────
            _err("")
            username = _prompt("CAS username:")
            page.locator("[name='identifier']").fill(username)
            page.locator("[name='identifier']").press("Enter")
            # Keep username in scope so we can persist it on successful auth.

            # ── Step 2b: Password (retry loop) ────────────────────────────
            # Wait for the password field to appear, then keep a locator
            # reference so Enter is pressed directly on the element (more
            # reliable than page.keyboard.press which depends on focus state).
            page.wait_for_selector(
                "[name='credentials.passcode']", state="visible", timeout=10_000
            )
            passwd_field = page.locator("[name='credentials.passcode']")
            _err("")

            while True:
                password = _prompt_password("CAS password (hidden):")
                passwd_field.fill(password)
                # Press Enter directly on the field — this triggers Okta's
                # form submit and initiates the push notification.
                passwd_field.press("Enter")

                # Give the page 5 s to show an inline error or proceed.
                # If no error appears, credentials were accepted.
                try:
                    page.wait_for_selector(
                        ".o-form-error-container:visible",
                        timeout=5_000,
                    )
                    # Error element appeared — wrong password
                    _error("Invalid password. Please try again.")
                    passwd_field.fill("")
                except PWTimeout:
                    # No error appeared → credentials accepted → exit loop
                    break

            # ── Step 3: Factor selection — choose Push notification ──────
            # After a correct password Okta shows a factor chooser; the user
            # must have at least "Recevoir une notification Push" available.
            # We click it automatically; if the element isn't found the user
            # may have TOTP-only and will need to handle that path manually.
            _step("Waiting for factor selection page…")
            page.wait_for_selector(
                '[data-se="okta_verify-push"]', state="visible", timeout=10_000
            )
            _step("Selecting Push notification…")
            page.locator('[data-se="okta_verify-push"] a').click()
            _ok("Push notification selected")

            # ── Step 3b: Number challenge (Okta may require picking a number) ──
            # After clicking Push, Okta often shows a 2-digit number that must
            # be selected on the phone. Wait briefly then try to extract it.
            _step("Checking for number challenge…")
            NUMBER_SELECTORS = [
                '[data-se="number-challenge-view"] .number',
                '[data-se="number-challenge-view"]',
                ".number-challenge-section",
                '[data-se="number-challenge"]',
            ]
            challenge_number = None
            for sel in NUMBER_SELECTORS:
                try:
                    page.wait_for_selector(sel, state="visible", timeout=3_000)
                    raw = page.locator(sel).first.inner_text().strip()
                    # The section contains a full sentence; extract just the 2-digit number.
                    m = re.search(r"\b(\d{2})\b", raw)
                    if m:
                        challenge_number = m.group(1)
                    break
                except PWTimeout:
                    pass

            if challenge_number is None:
                # Fallback: scan for any leaf element whose text is exactly 2 digits
                try:
                    found = page.evaluate("""
                        () => {
                            const els = Array.from(document.querySelectorAll('*'));
                            const leaf = els.find(e =>
                                e.children.length === 0 &&
                                /^\\d{2}$/.test(e.textContent.trim())
                            );
                            return leaf ? leaf.textContent.trim() : null;
                        }
                    """)
                    if found:
                        challenge_number = found
                except Exception:
                    pass

            # ── Step 4: MFA push notification ────────────────────────────
            _err("")
            _err(
                f"  {_BOLD}{_AMBER}╔══════════════════════════════════════════════════════╗{_RESET}"
            )
            _err(
                f"  {_BOLD}{_AMBER}║  📱  Check your phone — approve the Okta push        ║{_RESET}"
            )
            if challenge_number:
                _err(
                    f"  {_BOLD}{_AMBER}║                                                      ║{_RESET}"
                )
                _err(
                    f"  {_BOLD}{_AMBER}║  Select this number on your phone:  {_BOLD}{challenge_number:<17}{_AMBER}║{_RESET}"
                )
            _err(
                f"  {_BOLD}{_AMBER}║      Waiting up to 90 seconds…                       ║{_RESET}"
            )
            _err(
                f"  {_BOLD}{_AMBER}╚══════════════════════════════════════════════════════╝{_RESET}"
            )
            _err("")

            page.wait_for_function(
                "document.title.includes('SSL VPN Service')",
                timeout=90_000,
            )

            # ── Step 5: Extract the cookie ────────────────────────────────
            cookies = context.cookies()
            webvpn = next((c["value"] for c in cookies if c["name"] == "webvpn"), None)
            if webvpn is None:
                raise RuntimeError(
                    "Authentication appeared to succeed but no 'webvpn' "
                    "cookie was found. The VPN portal may have changed."
                )

            # Persist username so connect-vpn.sh / terminal-header.sh can
            # display it without re-prompting.
            try:
                with open(VPN_STATE_FILE, "w") as _f:
                    _f.write(username)
            except OSError:
                pass

            _ok("VPN cookie obtained successfully")
            _err("")
            return webvpn

        except PWTimeout as exc:
            _err("")
            _error(f"Timed out waiting for a page step: {exc}")
            raise
        except KeyboardInterrupt:
            _err("")
            _warn("Cancelled by user.")
            raise
        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    try:
        _banner()
        cookie = fetch_vpn_cookie()
        # Cookie → stdout ONLY (no newline, no colour — bash $() captures it cleanly)
        sys.stdout.write(cookie)
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        _error(f"VPN cookie fetch failed: {exc}")
        sys.exit(1)
