# Overnight run wrapper. Called by the Windows scheduled task every 5 hours.
#
# Every run is stateless: it reads the repository, does one experiment, commits,
# and exits. Nothing is carried in memory between runs. That is deliberate -- if a
# run dies partway through, whether from a usage limit, a crash, or a reboot, the
# next one reads the same files and carries on. There is nothing to resume.

$ErrorActionPreference = "Stop"
$repo = "C:\dev\aichessathon\starter"
$logs = Join-Path $repo "overnight\logs"
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$log = Join-Path $logs "$stamp.log"

New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $repo

"=== run $stamp ===" | Out-File -FilePath $log -Encoding utf8

# Refuse to start if the tree is dirty: a previous run died mid-edit and the
# champion may be broken. Better to stop and let a human look than to build on it.
$dirty = git status --porcelain
if ($dirty) {
    "ABORT: working tree is dirty, a previous run likely died mid-edit:" |
        Out-File -FilePath $log -Append -Encoding utf8
    $dirty | Out-File -FilePath $log -Append -Encoding utf8
    exit 0
}

$claude = "$env:USERPROFILE\.local\bin\claude.exe"
$prompt = Get-Content (Join-Path $repo "overnight\PROMPT.md") -Raw

$started = Get-Date
try {
    # --print runs headless and exits. Permissions are pre-granted in
    # .claude/settings.json so the run never blocks on an approval prompt that
    # nobody is awake to answer.
    $prompt | & $claude --print --permission-mode acceptEdits 2>&1 |
        Out-File -FilePath $log -Append -Encoding utf8
    $code = $LASTEXITCODE
} catch {
    $_ | Out-File -FilePath $log -Append -Encoding utf8
    $code = 1
}
$elapsed = [int]((Get-Date) - $started).TotalMinutes

"=== exit $code after $elapsed min ===" | Out-File -FilePath $log -Append -Encoding utf8

# A non-zero exit is usually a usage limit and is not an error worth alarming
# about: the next scheduled run picks up from the committed state. Exit 0 so the
# task history stays readable and Windows does not start backing the task off.
if ($code -ne 0) {
    "note: run ended early (exit $code). State is on disk; the next run continues." |
        Out-File -FilePath $log -Append -Encoding utf8
}

# Keep the last 60 logs, roughly a fortnight at four runs a day.
Get-ChildItem $logs -Filter "*.log" | Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 60 | Remove-Item -Force -ErrorAction SilentlyContinue

exit 0
