#!/bin/sh
# MittiGrid v0.4 — one-shot phone onboarding for Termux (Android).
#
# What this script does, in plain words:
#   1. asks for the coordinator's address (the laptop running MittiGrid)
#   2. installs Node.js, then builds llama.cpp's rpc-server on this phone
#      (the slow part: 15-40 minutes on a mid-range phone — a one-time cost,
#      re-running this script skips the build)
#   3. starts the rpc-server in the background
#   4. starts this phone's MittiGrid agent, pointed at the coordinator
#
# Run it from the mittigrid folder copied onto the phone:
#   sh scripts/onboard-phone.sh
#
# Keep the phone plugged in. A low battery makes the phone join in standby
# on purpose (it recovers on charge) and a hot phone slows itself down.

set -e

line() { echo "------------------------------------------------------------"; }

line
echo " MITTIGRID PHONE SETUP"
line
echo ""
echo "This turns this phone into one worker in your MittiGrid."
echo "The first run takes a while (building: 15-40 minutes on a"
echo "mid-range phone). It is mostly automatic — you will only be"
echo "asked one question. Plug the phone in now if you can."
echo ""

# ---- [1/6] find the repo files ---------------------------------------------
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname -- "$SCRIPT_DIR")
if [ ! -f "$REPO_DIR/agent.js" ]; then
  echo "STOP: I cannot find agent.js one folder above this script."
  echo "I looked at: $REPO_DIR/agent.js"
  echo ""
  echo "How to fix: copy the whole mittigrid folder onto the phone (for"
  echo "example into the Home folder of Termux), then run this script"
  echo "from inside it:"
  echo "  sh scripts/onboard-phone.sh"
  exit 1
fi
echo "[1/6] mittigrid folder found at: $REPO_DIR"

# ---- [2/6] ask for the coordinator's address -------------------------------
# An IP address is a device's street address on your home Wi-Fi. The
# coordinator's address is the laptop's IP plus the port 7400.
SAVED="$HOME/.mittigrid-coordinator-url"
DEFAULT_COORD=""
if [ -f "$SAVED" ]; then DEFAULT_COORD=$(cat "$SAVED" 2>/dev/null); fi
echo ""
echo "[2/6] I need the address of the coordinator (the laptop running"
echo "      MittiGrid). On the laptop the dashboard opens at"
echo "      http://localhost:7400 — but from THIS phone you need the"
echo "      laptop's IP instead, like http://192.168.1.5:7400"
echo "      (On the laptop, run: ipconfig  — and look for 'IPv4 Address'.)"
printf "Coordinator address%s: " "${DEFAULT_COORD:+ [press Enter to reuse $DEFAULT_COORD]}"
read -r COORD
[ -z "$COORD" ] && COORD="$DEFAULT_COORD"
if [ -z "$COORD" ]; then
  echo "STOP: I need an address to continue. Run me again and type one."
  exit 1
fi
# Be forgiving: add the http:// and the :7400 port if the user left them off.
case "$COORD" in
  http://*|https://*) : ;;
  *) COORD="http://$COORD" ;;
esac
case "$COORD" in
  *:[0-9]*) : ;;
  *) COORD="$COORD:7400" ;;
esac
COORD=${COORD%/}
mkdir -p "$(dirname "$SAVED")" 2>/dev/null || true
echo "$COORD" > "$SAVED" 2>/dev/null || true
echo "      -> will use: $COORD (saved, so next time just press Enter)"

# ---- [3/6] install Node.js --------------------------------------------------
echo ""
echo "[3/6] installing Node.js (it runs the MittiGrid agent on this"
echo "      phone). One line: Node.js is a program that runs JavaScript."
pkg install -y nodejs > /dev/null 2>&1 || pkg install -y nodejs
echo "      installed: node $(node --version)"

# ---- [4/6] check the coordinator is reachable (fail fast, before the build) --
echo ""
echo "[4/6] checking the coordinator answers at $COORD"
echo "      (waiting up to 5 seconds)"
if node -e "fetch('$COORD/status.json',{signal:AbortSignal.timeout(5000)}).then(r=>{if(!r.ok)throw 0;return r.json()}).then(s=>{console.log('      reachable. agents online right now: '+(s.stats?s.stats.online:'?'))}).catch(()=>{console.log('      NOT reachable');process.exit(1)})"; then :; else
  echo ""
  echo "STOP: the coordinator did not answer at $COORD"
  echo "Check, in this order:"
  echo "  1. Is 'node coordinator.js' running on the laptop?"
  echo "  2. Are the phone and the laptop on the same Wi-Fi network?"
  echo "     (Phones often drop Wi-Fi when the screen is off — wake the"
  echo "     phone and check the Wi-Fi symbol.)"
  echo "  3. Is the IP right? On the laptop run: ipconfig  (IPv4 Address)"
  echo ""
  echo "Nothing important was changed yet — fix the problem and run me"
  echo "again. The address you typed is saved; just press Enter next time."
  exit 1
fi

# ---- [5/6] build llama.cpp's rpc-server (the slow step) ---------------------
echo ""
echo "[5/6] building llama.cpp's rpc-server for this phone."
echo "      This is the slow step: 15-40 minutes on a mid-range phone,"
echo "      longer on an old one. A lot of text will scroll by — that is"
echo "      normal, you can walk away. If it was built before, this is quick."
if [ -x "$REPO_DIR/build/bin/rpc-server" ]; then
  echo "      already built: $REPO_DIR/build/bin/rpc-server — reusing it"
else
  (cd "$REPO_DIR" && sh scripts/build-rpc-termux.sh)
fi

# ---- [6/6] start rpc-server, then join the grid ------------------------------
echo ""
echo "[6/6] starting rpc-server in the background (port 50052)."
echo "      One line: rpc-server shares this phone's memory with the"
echo "      laptop, so together they can hold a bigger AI model."
if command -v pkill > /dev/null 2>&1; then
  pkill rpc-server 2>/dev/null || true
fi
"$REPO_DIR/build/bin/rpc-server" --host 0.0.0.0 --port 50052 > "$HOME/rpc-server.log" 2>&1 &
RPC_PID=$!
sleep 2
if ! kill -0 "$RPC_PID" 2>/dev/null; then
  echo "STOP: rpc-server did not start. Last lines of its log ($HOME/rpc-server.log):"
  tail -5 "$HOME/rpc-server.log" 2>/dev/null || true
  echo "Run me again after checking the error above."
  exit 1
fi
echo "      running (pid $RPC_PID, log: ~/rpc-server.log)."
if command -v pkill > /dev/null 2>&1; then
  echo "      to stop it later, from any Termux session: pkill rpc-server"
else
  echo "      to stop it later: kill $RPC_PID"
fi

if command -v termux-wake-lock > /dev/null 2>&1; then
  termux-wake-lock 2>/dev/null || true
  echo "      wake lock held (stops Android from dozing and pausing the"
  echo "      work; release later with: termux-wake-unlock)"
fi

echo ""
line
echo " JOINING THE GRID — this stays running until you press Ctrl+C"
line
echo ""
echo "WHAT TO EXPECT:"
echo "  - lines starting with [mittigrid] are this phone's agent talking."
echo "  - 'battery NN%' — if it says low, plug the phone in: it joins"
echo "    fully once charging (a designed rule, not a bug)."
echo "  - 'rpc-server advertised' — the laptop can now use this phone's"
echo "    memory when it builds its llama-server command."
echo "  - 'shard assigned' — this phone now hosts part of the model."
echo "  - On the laptop's dashboard you should see this phone under"
echo "    AGENTS and REAL MODEL within a few seconds."
echo "  - Then chat at:  http://<laptop-ip>:7400/chat"
echo ""
export MITTI_RPC_PORT=50052
exec node "$REPO_DIR/agent.js" --coord "$COORD"
