# Register the one-shot night run and pause the five-hourly Claude loop for it.
#
#   .\overnight\install-night.ps1                 # start in 2 minutes
#   .\overnight\install-night.ps1 -At "23:45"     # start at a clock time tonight
#
# The night script needs the whole machine, so the Claude loop is disabled here and
# must be re-enabled by hand in the morning:
#   Enable-ScheduledTask -TaskName "AIChessathon-Overnight"
#
# Remove the night task with:  Unregister-ScheduledTask -TaskName "AIChessathon-Night"

param([string]$At = "", [int]$InMinutes = 2)

$ErrorActionPreference = "Stop"
$name = "AIChessathon-Night"
$bash = "C:\Program Files\Git\bin\bash.exe"
$script = "C:\dev\aichessathon\starter\overnight\night.sh"

if (-not (Test-Path $bash)) { throw "missing $bash" }
if (-not (Test-Path $script)) { throw "missing $script" }

$action = New-ScheduledTaskAction -Execute $bash `
    -Argument "-lc `"cd /c/dev/aichessathon/starter && ./overnight/night.sh >> overnight/night.task.out 2>&1`""

if ($At) {
    $start = [datetime]::ParseExact($At, "HH:mm", $null)
    if ($start -lt (Get-Date)) { $start = $start.AddDays(1) }
} else {
    $start = (Get-Date).AddMinutes($InMinutes)
}
$trigger = New-ScheduledTaskTrigger -Once -At $start

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 14) `
    -MultipleInstances IgnoreNew

Disable-ScheduledTask -TaskName "AIChessathon-Overnight" -ErrorAction SilentlyContinue | Out-Null
Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Description "AI Chessathon: one gated night of experiments." | Out-Null

$info = Get-ScheduledTaskInfo -TaskName $name
Write-Host "registered '$name'"
Write-Host "  next run : $($info.NextRunTime)"
Write-Host "  claude loop 'AIChessathon-Overnight' : $((Get-ScheduledTask -TaskName 'AIChessathon-Overnight').State)"
Write-Host ""
Write-Host "Leave the laptop plugged in and logged on (locking the screen is fine)."
Write-Host "Progress: overnight\night.task.out, overnight\night\night.log, overnight\night\SUMMARY.md"
