# Register (or re-register) the overnight scheduled task.
#
# Windows Task Scheduler rather than a Claude-session cron, because those live only
# as long as the session that made them and only fire while the REPL is idle. This
# runs whether or not the Claude app is open, and survives a reboot.
#
#   .\.venv\Scripts\python.exe  ->  overnight\run.ps1  ->  claude --print
#
# Remove with:  Unregister-ScheduledTask -TaskName "AIChessathon-Overnight"
# Pause with:   Disable-ScheduledTask   -TaskName "AIChessathon-Overnight"
# Run now with: Start-ScheduledTask     -TaskName "AIChessathon-Overnight"
#
# -FirstRunInMinutes sets when the first run fires; the 5-hour repetition follows
# from there. Point it just after a usage-limit reset: a run that starts with the
# quota nearly spent dies partway through an experiment and wastes the slot.

param([int]$FirstRunInMinutes = 10)

$ErrorActionPreference = "Stop"
$name = "AIChessathon-Overnight"
$script = "C:\dev\aichessathon\starter\overnight\run.ps1"

if (-not (Test-Path $script)) { throw "missing $script" }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`""

# Fire every 5 hours, which is the subscription's rolling usage window. Anchoring
# the first run just after a reset keeps every later run landing near a fresh
# window too, since the cadence and the window are the same length.
$first = (Get-Date).AddMinutes($FirstRunInMinutes)
$trigger = New-ScheduledTaskTrigger -Once -At $first `
    -RepetitionInterval (New-TimeSpan -Hours 5)

# WakeToRun matters: a sleeping laptop runs nothing. StartWhenAvailable catches up
# a run missed while the machine was off. The 4h30m kill stops a wedged run from
# colliding with the next one.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4 -Minutes 30) `
    -MultipleInstances IgnoreNew

Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Description "AI Chessathon: one engine experiment every 5 hours." | Out-Null

$task = Get-ScheduledTask -TaskName $name
$info = Get-ScheduledTaskInfo -TaskName $name
Write-Host "registered '$name'"
Write-Host "  state    : $($task.State)"
Write-Host "  next run : $($info.NextRunTime)"
Write-Host "  interval : every 5 hours"
Write-Host ""
Write-Host "The machine must be awake and plugged in. Check Settings > System >"
Write-Host "Power for a sleep timer that would stop it; WakeToRun only helps if"
Write-Host "the machine is asleep, not if it is shut down."
