# Manual E2E Checklist

Run these on Windows 10 and Windows 11 before cutting a release.

## First-Run Calibration

1. Install source or the packaged app.
2. Install Ollama and pull the configured vision model.
3. Launch `python -m lolcoach` and confirm the desktop control panel opens.
4. Use Setup health checks to confirm missing dependencies are shown clearly.
5. Start League, enter Practice Tool as jungle, and run calibration from Setup.
6. Click the minimap top-left and bottom-right corners.
7. Confirm `calibration.json` is written and Start enables the coach.
8. Expected: one coherent voice callout within 3 minutes.

## Non-Jungle Role

1. Start a game as a non-jungle role.
2. Run the coach.
3. Expected: exactly one "You're not playing jungle. Coach paused." line.
4. Expected: no inference calls until the next game.

## Multi-Game Session

1. Play or simulate three games back-to-back.
2. Confirm a fresh JSONL log file is created per game.
3. Confirm state from game 1 does not appear in game 2 or 3 callouts.

## Match-Start Auto Launch

1. Launch the desktop control panel on Windows.
2. Enable `Launch lolcoach when a match starts` in Settings.
3. Log out and back in, or confirm the monitor starts immediately after enabling.
4. Start a Practice Tool jungle game.
5. Expected: lolcoach appears only in the tray and starts coaching automatically.
6. Exit the game and start another match.
7. Expected: one new launch attempt for the new match, not repeated launches during loading.
8. Disable the setting.
9. Expected: the Scheduled Task is removed and future matches do not launch lolcoach.

## Remake / New Game

1. During an active game, simulate a new Live Client payload whose `gameTime`
   jumps backward by more than 30 seconds.
2. Expected: lifecycle cycles through ending/idle/starting and state resets.

## GPU Contention

1. Run League and OBS/ShadowPlay.
2. Run `python -m lolcoach.benchmark --fixture tests/fixtures/minimap_sample.png`.
3. Expected: p95 latency is inside the chosen model path from the README
   decision tree.

## Packaging Smoke

1. Build with `python scripts/build-exe.py`.
2. Build installer with `makensis scripts/build-installer.nsi`.
3. Install on clean Windows 10 and Windows 11 VMs.
4. Expected: installed app opens the desktop control panel, calibrates, reads
   config, speaks audio, and writes JSONL logs.
