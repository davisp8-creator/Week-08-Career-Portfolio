#!/usr/bin/env bash
#=============================================================================
#region SCRIPT DETAILS
#=============================================================================
#.SYNOPSIS
#Checks for macOS, verifies prerequisites, and installs the required Python
#packages for the positional-encoding ablation into a local .venv.
#
#.DESCRIPTION
#- Verifies Python 3.10+ is installed
#- Creates (or reuses) a project-local virtual environment in .venv
#  (avoids Homebrew's "externally-managed-environment" pip error)
#- Upgrades pip inside the venv
#- Installs the pinned local environment (requirements-cpu.txt plus the pinned CPU torch)
#- Runs env_setup.py, which checks the imports of the entry point and reports the compute device
#- Offers a quick smoke test or the full run after setup
#- Reports Apple silicon vs Intel Mac and whether this Mac is itself a VM (env_setup.py)
#
#.NOTES
#Double-click in Finder, or run:  bash Install-macOS.command
#If macOS says it cannot be opened: right-click > Open, or run  chmod +x <this file>  once.
#Unattended:  LAUNCH=none bash Install-macOS.command
#Written for the bash 3.2 that ships with macOS (no associative arrays).
#=============================================================================
#endregion
#=============================================================================
#region Prerequisites
#=============================================================================
#  OS CHECK: Only run on macOS (Darwin)
OS_NAME=$(uname)
if [ "$OS_NAME" != "Darwin" ]; then
    echo "OS is not macOS. This script is only intended for macOS devices."
    exit 666
fi
#=============================================================================
#endregion
#=============================================================================
#region FUNCTIONS
#=============================================================================

set -eo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m';  GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m';     RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}$*${RESET}\n"; }
pause_exit() {
    if [ -z "${LAUNCH:-}" ]; then read -rp "Press Enter to exit..." _; fi
    exit "${1:-1}"
}

# ── Change to the script's own directory ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENTRY_SCRIPT="scripts/reproduce.py"
ENV_SETUP="env_setup.py"
REQUIREMENTS="requirements-cpu.txt"
TORCH_PIN="torch==2.14.1"
VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"
MIN_MAJOR=3
MIN_MINOR=10

[ -z "${LAUNCH:-}" ] && clear
echo -e "${BOLD}=================================================================${RESET}"
echo -e "${BOLD}   Positional-Encoding Ablation - macOS Installer${RESET}"
echo -e "${BOLD}=================================================================${RESET}"
echo ""
#=============================================================================
#endregion
#=============================================================================
#region EXECUTION
#=============================================================================
#Step 1: Find Python 3.10+
#=============================================================================
header "Step 1: Locating Python ${MIN_MAJOR}.${MIN_MINOR}+"

PYTHON=""
for cmd in python3 python3.14 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$cmd" >/dev/null 2>&1 && \
       "$cmd" -c "import sys; sys.exit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null; then
        PYTHON="$cmd"
        PY_VER=$("$PYTHON" -c "import sys; print(sys.version.split()[0])")
        success "Found: $PYTHON  (Python $PY_VER)"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python ${MIN_MAJOR}.${MIN_MINOR}+ was not found on this system."
    echo ""
    echo -e "  ${BOLD}Install Python from one of these sources:${RESET}"
    echo "    • Official:  https://www.python.org/downloads/"
    echo "    • Homebrew:  brew install python"
    echo ""
    echo "  After installing Python, re-run this script."
    pause_exit 1
fi

if [ "$(uname -m)" = "x86_64" ]; then
    warn "Intel Mac detected. Recent PyTorch releases ship macOS wheels for Apple Silicon only,"
    warn "so the torch install may fail here. If it does, run on a Colab T4 instead (requirements-t4.txt)."
fi

#=============================================================================
#Step 2: Check application files
#=============================================================================
header "Step 2: Checking application files"

for f in "$ENTRY_SCRIPT" "$ENV_SETUP" "$REQUIREMENTS"; do
    if [ -f "$f" ]; then
        success "Found: $f"
    else
        error "Not found: $f"
        warn "Run this installer from the project folder (next to env_setup.py)."
        pause_exit 1
    fi
done

#=============================================================================
#Step 3: Create / reuse the virtual environment
#=============================================================================
header "Step 3: Preparing virtual environment (.venv)"

if [ -x "$VENV_PY" ]; then
    success "Existing .venv found - reusing it"
else
    "$PYTHON" -m venv "$VENV_DIR" || { error "Failed to create .venv"; pause_exit 1; }
    success "Created .venv"
fi

#=============================================================================
#Step 4: Upgrade pip
#=============================================================================
header "Step 4: Ensuring pip is up to date"

"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
if ! "$VENV_PY" -m pip install --upgrade pip --quiet; then
    error "pip upgrade failed."
    warn "With python.org Python, an SSL error usually means you need to run"
    warn "'Install Certificates.command' from /Applications/Python 3.x/ once."
    pause_exit 1
fi
success "pip is ready"

#=============================================================================
#Step 5: Install the pinned environment, then let env_setup.py check the imports
#=============================================================================
header "Step 5: Installing pinned packages"
info "This can take a few minutes the first time: torch is large."

# env_setup.py refuses unpinned installs, so install the versions that produced results/*_cpu.* first.
# The exact pins exist only for the Python version they were recorded on (3.14); on another version fall back
# to whatever pip resolves, and say so, because the numbers may then differ in the last digits.
if ! "$VENV_PY" -m pip install --quiet -r "$REQUIREMENTS" "$TORCH_PIN"; then
    warn "The pinned versions could not be installed on this Python. Falling back to unpinned packages."
    warn "Results will record the versions actually used in results/manifest_cpu.json."
    export ALLOW_UNPINNED_INSTALL=1
fi

if ! "$VENV_PY" "$ENV_SETUP"; then
    error "env_setup.py failed - see the messages above."
    pause_exit 1
fi

#=============================================================================
#Step 6: Verify imports
#=============================================================================
header "Step 6: Verifying imports"

VERIFY_FAILED=0
for import_name in torch numpy pandas matplotlib scipy yaml psutil; do
    if "$VENV_PY" -c "import $import_name" >/dev/null 2>&1; then
        success "$import_name successfully imported"
    else
        error "Failed to import: $import_name"
        VERIFY_FAILED=1
    fi
done

if [ "$VERIFY_FAILED" -eq 1 ]; then
    echo ""
    error "One or more packages failed verification."
    pause_exit 1
fi

info "Macs have no NVIDIA GPU, so the local run uses the CPU (Apple MPS is detected but not used: one reference"
info "device per tag keeps the arms comparable). The T4 reference run was done on Colab (requirements-t4.txt)."

#=============================================================================
#Step 7: Launch prompt
#=============================================================================
echo ""
echo -e "${BOLD}=================================================================${RESET}"
echo -e "${BOLD}   Setup Complete${RESET}"
echo -e "${BOLD}=================================================================${RESET}"
echo ""

CHOICE="${LAUNCH:-}"
if [ -z "$CHOICE" ]; then
    echo -e "  ${BOLD}1${RESET}  Quick smoke test (about 10 s on CPU, tiny budget - numbers are meaningless)"
    echo -e "  ${BOLD}2${RESET}  Full run (regenerates the cpu tables and plots from results/runs_cpu.csv in about a minute)"
    echo -e "  ${BOLD}3${RESET}  Exit without running"
    echo ""
    while true; do
        read -rp "  Enter your choice [1/2/3]: " c
        case "$c" in
            1) CHOICE=quick; break ;;
            2) CHOICE=full;  break ;;
            3) CHOICE=none;  break ;;
            *) warn "Please enter 1, 2 or 3." ;;
        esac
    done
fi

case "$CHOICE" in
    quick)
        info "Running quick smoke test ..."
        ABLATE_QUICK=1 "$VENV_PY" "$ENTRY_SCRIPT"
        ;;
    full)
        info "Running the full entry point (resumes from results/runs_cpu.csv when present) ..."
        "$VENV_PY" "$ENTRY_SCRIPT"
        ;;
    *)
        info "To run manually:"
        echo "    .venv/bin/python $ENTRY_SCRIPT"
        info "Retrain every run from scratch (about 1 h on CPU):"
        echo "    ABLATE_FRESH=1 .venv/bin/python $ENTRY_SCRIPT"
        info "Repeat the whole study:"
        echo "    .venv/bin/python scripts/repeat_study.py --n 3"
        info "Quick smoke test:"
        echo "    ABLATE_QUICK=1 .venv/bin/python $ENTRY_SCRIPT"
        ;;
esac

echo ""
#=============================================================================
#endregion
#=============================================================================
