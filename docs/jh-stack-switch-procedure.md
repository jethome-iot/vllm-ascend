# Stack switch procedure: MindIE ↔ vllm

**Purpose:** Safely switch between MindIE and vllm-ascend на 10.183.1.25.
Both share `/dev/davinci*` (physical constraint per AGENTS.md §«НЕ работали» 6 —
«vllm и MindIE не запускать одновременно»).

**Audience:** Engineer (adeep) executing SP-0b baseline regression + SP-1+ NPU tests.

**Constraints:**
- AGENTS.md hard rules: TP ≤ 2 single-card; `restart: "no"` for all inference containers
- Dev server (per user 2026-05-29): no users to notify; switching freely
- IPMI accessible (per user 2026-05-29): hard reset available если что-то застряло

---

## Pre-switch verification (mandatory before each risky switch)

```bash
# Verify restart policy на all profiles — defense against AGENTS.md §«НЕ работали» 3
ssh adeep@10.183.1.25 'grep -A2 restart: /opt/vllm/docker-compose.yml /opt/mindie/docker-compose.yml | grep restart:'
# Expected output: ALL lines should show 'restart: "no"'
# If anywhere `unless-stopped` или `always` — STOP, fix first, then proceed
```

---

## MindIE → vllm

```bash
# 1. Stop MindIE
ssh adeep@10.183.1.25 '/opt/mindie/mindie.sh stop'

# 2. Wait for NPU чипы to clear (timeout 5 min)
ssh adeep@10.183.1.25 'until sudo /usr/local/sbin/npu-smi info | grep -q "No running"; do sleep 5; done; echo "NPU clear"'

# 3. Verify ALL 4 chips free
ssh adeep@10.183.1.25 'sudo /usr/local/sbin/npu-smi info | grep "Process id"'
# Expected: each chip shows "No running process"

# 4. Start vllm with desired profile
ssh adeep@10.183.1.25 '/opt/vllm/vllm.sh <profile-name>'

# 5. Wait for endpoint up (timeout 5 min)
ssh adeep@10.183.1.25 'until curl -s :8000/v1/models 2>/dev/null | grep -q .; do sleep 5; done; echo "vllm UP"'
```

---

## vllm → MindIE

```bash
# 1. Stop vllm
ssh adeep@10.183.1.25 '/opt/vllm/vllm.sh stop'

# 2. Wait for NPU clear
ssh adeep@10.183.1.25 'until sudo /usr/local/sbin/npu-smi info | grep -q "No running"; do sleep 5; done'

# 3. Start MindIE с baseline profile (qwen32)
ssh adeep@10.183.1.25 '/opt/mindie/mindie.sh qwen32'

# 4. Wait for endpoint up
ssh adeep@10.183.1.25 'until curl -s :1025/v1/models 2>/dev/null | grep -q qwen3-32b; do sleep 5; done; echo "MindIE UP"'

# 5. Verify production smoke (1 prompt)
ssh adeep@10.183.1.25 'curl -s :1025/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-32b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":50}" | head -c 200'

# 6. ⚠️ RESTORE open-webui (gotcha — см. ниже). vllm.sh stop / compose --profile stop
#    гасит open-webui (тот же compose-проект /opt/vllm, no-profile, но stop его ловит).
#    MindIE — отдельный проект, обратно его НЕ поднимает.
ssh adeep@10.183.1.25 'cd /opt/vllm && sudo docker compose up -d open-webui'
ssh adeep@10.183.1.25 'until [ "$(curl -s -o /dev/null -w %{http_code} http://localhost:8080/health)" = "200" ]; do sleep 5; done; echo "open-webui UP"'
# Первый старт после recreate ~1-2 мин (грузит embedding all-MiniLM-L6-v2 в память). HTTP 000 в это окно — норма.
```

## ⚠️ GOTCHA: open-webui гасится при остановке vllm-профиля

`/opt/vllm/vllm.sh stop` (и `docker compose --profile <X> stop`) останавливают НЕ
только vllm-контейнер, но и `open-webui` — он в том же compose-проекте `/opt/vllm`,
и хотя объявлен без профиля (always-on), `stop` его всё равно ловит. Рестарт MindIE
(`/opt/mindie/mindie.sh` — **отдельный** compose-проект) open-webui обратно НЕ
поднимает.

**Следствие:** после каждого цикла «vllm test → restore MindIE» open-webui остаётся
stopped, UI на :8080 не отвечает. **Fix:** шаг 6 выше (всегда после restore MindIE).

Обнаружено 2026-05-31 (SP-6 spike).

---

## Pre-launch watchdog (mandatory before A2+ risky launches)

Per master plan §14.4 «Kernel hang debugging procedure». Captures pre-hang state if NPU crashes.

```bash
# 1. Create watchdog log dir, clean prior runs
ssh adeep@10.183.1.25 'mkdir -p /var/log/jh-watchdog && rm -f /var/log/jh-watchdog/*.log'

# 2. Start npu-smi stream в фоне
ssh adeep@10.183.1.25 'nohup bash -c "while true; do date +%s >> /var/log/jh-watchdog/npu-smi.log; sudo /usr/local/sbin/npu-smi info >> /var/log/jh-watchdog/npu-smi.log 2>&1; sleep 2; done" &> /dev/null &'

# 3. Start dmesg stream
ssh adeep@10.183.1.25 'nohup sudo dmesg -W > /var/log/jh-watchdog/dmesg-stream.log 2>&1 &'

# 4. Write start marker
ssh adeep@10.183.1.25 "echo '=== START $(date) workload=<workload-name> ===' > /var/log/jh-watchdog/markers.log"

# (later) start container log capture после container exists
ssh adeep@10.183.1.25 'docker logs -f <container-name> > /var/log/jh-watchdog/container.log 2>&1 &'
```

### Watchdog cancellation (after safe completion)

```bash
# Kill background watchdog processes
ssh adeep@10.183.1.25 'pkill -f "npu-smi.log" 2>/dev/null; pkill -f "dmesg -W" 2>/dev/null; pkill -f "docker logs -f" 2>/dev/null'

# Archive logs
ssh adeep@10.183.1.25 "tar -czf /tmp/jh-watchdog-$(date +%s).tar.gz /var/log/jh-watchdog/"

# Cleanup (optional — keep archive)
ssh adeep@10.183.1.25 'rm -rf /var/log/jh-watchdog/*.log'
```

---

## Recovery if NPU hangs (per master §14.4)

### Case A: Host responsive, npu-smi responds

```bash
# 1. Capture state
ssh adeep@10.183.1.25 'sudo /usr/local/sbin/npu-smi info > /tmp/hang-$(date +%s).txt; \
  dmesg | tail -200 >> /tmp/hang-$(date +%s).txt'

# 2. Stop offending container
ssh adeep@10.183.1.25 'docker rm -f <container-name>'

# 3. Wait clear, then proceed normal recovery
```

### Case B: Host responsive, npu-smi hangs

```bash
# Soft reboot acceptable since host responds
ssh adeep@10.183.1.25 'sudo reboot'

# Wait for boot
until ssh -o ConnectTimeout=5 adeep@10.183.1.25 'uptime' 2>/dev/null; do sleep 5; done
echo "Host UP"
```

### Case C: Host frozen / SSH unresponsive (most likely для cross-card TP=4 hang)

**Use IPMI** (per master P2.2 v3, AGENTS.md updated 2026-05-29):

1. Connect to IPMI console (URL/creds in user's password manager)
2. Hard reset через IPMI control
3. Wait для boot, verify через SSH

After recovery:
- Verify `restart: "no"` honored (no container auto-started)
- Verify baseline Qwen3-32B regression
- Write investigation: `huawei-ascend/docs/investigations/<date>-kernel-hang-<phase>.md`
- Update `huawei-ascend/memory/atlas-300i-duo-tp-cross-card-kernel-hang.md` с новым случаем

---

## Switching cost reference

- Switch overhead: ~3-5 минут per direction (MindIE warm-restart ~2 min, vllm warm-restart ~30s)
- In SP-2+ iterative work — batch test runs во избежание excessive switching
- Watchdog setup overhead: ~30s start; archive ~10s
