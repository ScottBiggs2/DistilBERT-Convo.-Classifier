Run both labeling scripts in background (nohup)

Files added:
- `run_both.sh`  — starts both scripts with `nohup`, writes logs to `./logs/` and pid files.
- `stop_both.sh` — stops processes started by `run_both.sh` by reading pid files in `./logs/`.

Quick start (macOS zsh)

1) Make the scripts executable (one-time):

```bash
chmod +x run_both.sh stop_both.sh
```

2) (Optional) Activate your environment or use the included `.venv` if you created one.
The runner will automatically source `./.venv/bin/activate` if that file exists.

3) Start both scripts (detached):

```bash
./run_both.sh
```

This will:
- create `logs/` if not present
- start `src/get_gemini_flash_labels.py` -> `logs/gemini.out` and `logs/gemini.err` (pid -> `logs/gemini.pid`)
- start `src/get_gpt_4o_mini_logprobs_unverified.py` -> `logs/gpt4o_unverified.out` and `logs/gpt4o_unverified.err` (pid -> `logs/gpt4o_unverified.pid`)

Monitoring

- Tail logs in realtime:

```bash
# one log
tail -f logs/gemini.out

# both stdout and err
tail -f logs/gemini.out logs/gemini.err logs/gpt4o_unverified.out logs/gpt4o_unverified.err
```

- Check if processes are running:

```bash
ps -p $(cat logs/gemini.pid) -o pid,cmd
ps -p $(cat logs/gpt4o_unverified.pid) -o pid,cmd
```

Stopping

```bash
./stop_both.sh
```

This will attempt a graceful kill and fall back to SIGKILL if needed. It will also remove the pid files.

Notes & recommendations

- Ensure environment variables (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`) are available to the shell that runs `./run_both.sh`.
  If you keep keys in a `.env` file, load them before running, e.g.:

```bash
set -a; source .env; set +a
./run_both.sh
```

--

Automatically bypassing Gemini's interactive confirmation

The Gemini labeler included in `src/get_gemini_flash_labels.py` prompts for confirmation when it detects a large dataset. If you want `run_both.sh` to automatically answer `y` to that prompt (useful for unattended runs), set the env var `AUTO_CONFIRM_GEMINI=1` in the shell that launches the runner:

```bash
set -a; source .env; set +a
AUTO_CONFIRM_GEMINI=1 ./run_both.sh
```

This will pipe a single `y` into the Gemini script only at startup so it doesn't block on `input()`.

- The project `requirements.txt` lists many packages. If you hit a `ModuleNotFoundError` for `google-generativeai`, install it manually:

```bash
pip install google-generativeai
```

- If you prefer a supervision system (auto-restart on crash) or better logging rotation, consider `supervisord`, `launchd` (macOS), or running within `tmux`.

- If you'd like, I can also add a small Python supervisor that restarts on failure and writes structured JSON logs.
