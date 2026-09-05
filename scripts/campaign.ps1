# Unattended, resumable fault-injection campaign against the running OpenTelemetry Demo.
#
#   powershell -File scripts\campaign.ps1 -Chunks 20 -PerChunk 20 -BaseSeed 100
#
# Runs Chunks x PerChunk experiments (each chunk has its own seed so the plan is
# reproducible), truncating the collector's capture files between chunks so that per-
# experiment parsing stays fast. A chunk whose last experiment directory already exists
# is skipped, so re-running the same command resumes after an interruption. Every
# experiment is ~2 minutes with the timings below; 400 experiments is ~14 hours.
param(
    [int]$Chunks = 10,
    [int]$PerChunk = 20,
    [int]$BaseSeed = 100,
    [string]$OutRoot = "data\real",
    [string]$DemoDir = "bench\opentelemetry-demo",
    [double]$WarmupS = 60,
    [double]$FaultMinS = 35,
    [double]$FaultMaxS = 75,
    [double]$CooldownS = 25,
    # >= 30 s: a VU change restarts k6, and at 15 s the traffic gap landed inside the
    # recorded warm-up, which every detector (rules included) read as an incident --
    # ~34 false alarms/h on normal experiments at K=1.
    [double]$SettleS = 30,
    [double]$OffsetMaxS = 20,
    # Docker Desktop's per-user install does not put docker.exe on PATH. Point this at
    # the directory holding docker.exe if `docker` is not already resolvable; it is
    # appended to PATH only when it exists. Set to "" when docker is already on PATH.
    [string]$DockerBin = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin",
    # The `rca` entry point; defaults to the repo's own virtualenv.
    [string]$RcaExe = ""
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if ($DockerBin -and (Test-Path $DockerBin)) { $env:PATH = "$env:PATH;$DockerBin" }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker not found on PATH; pass -DockerBin <directory containing docker.exe>"
}
$rca = if ($RcaExe) { $RcaExe } else { Join-Path $repo ".venv\Scripts\rca.exe" }
if (-not (Test-Path $rca)) { throw "rca not found at $rca; pass -RcaExe <path to rca>" }
$compose = @("compose", "-f", "compose.yaml", "-f", "compose.full.yaml", "-f", "docker-compose.override.yml")
$capture = Join-Path $DemoDir "rca-capture"
$flags = Join-Path $DemoDir "src\flagd\demo.flagd.json"
New-Item -ItemType Directory -Force $OutRoot | Out-Null
$log = Join-Path $OutRoot "campaign.log"

# Every container compose.yaml + compose.full.yaml + the override bring up. The campaign
# is only meaningful if all of them are running, so this is asserted rather than assumed.
$expected = @(
    "accounting", "ad", "astronomy-db", "cart", "checkout", "currency", "email",
    "flagd", "flagd-ui", "fraud-detection", "frontend", "frontend-proxy",
    "image-provider", "kafka", "load-generator", "otel-collector", "payment",
    "product-catalog", "quote", "recommendation", "shipping", "telemetry-docs",
    "valkey-cart"
)

function Write-Log($msg) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $msg
    Write-Host $line
    # Out-File, not Tee-Object: Tee-Object writes UTF-16LE on Windows PowerShell 5.1, so
    # the log is unreadable to anything expecting text.
    $line | Out-File -FilePath $log -Encoding utf8 -Append
}

function Write-LogRaw($lines) {
    foreach ($line in $lines) {
        Write-Host $line
        $line | Out-File -FilePath $log -Encoding utf8 -Append
    }
}

function Invoke-Logged([string]$exe, [string[]]$argv) {
    # Start-Process keeps the child's stderr out of the PowerShell pipeline entirely.
    # With $ErrorActionPreference = "Stop", piping a native command through `2>&1 |`
    # promotes its first stderr line to a terminating error, so the exit-code check
    # below never runs and a chunk that merely logged a warning looks like a crash.
    # Two temp files because -RedirectStandardOutput and -RedirectStandardError may not
    # share one.
    $outFile = [System.IO.Path]::GetTempFileName()
    $errFile = [System.IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -Wait `
            -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        foreach ($f in @($outFile, $errFile)) {
            # -Encoding UTF8: the child writes UTF-8 and Get-Content would otherwise
            # decode it as the system ANSI codepage.
            if ((Get-Item $f).Length -gt 0) { Write-LogRaw (Get-Content $f -Encoding UTF8) }
        }
        return $proc.ExitCode
    } finally {
        Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue
    }
}

function Reset-Stack {
    $code = Invoke-Logged $rca @("bench", "reset", "--flags-path", $flags)
    if ($code -ne 0) { Write-Log "warning: rca bench reset exited $code" }
}

function Clear-Capture {
    Push-Location $DemoDir
    try {
        & docker @compose stop otel-collector | Out-Null
        foreach ($f in "traces", "metrics", "logs") {
            $p = Join-Path "rca-capture" "$f.jsonl"
            if (Test-Path $p) { Clear-Content $p }
        }
        & docker @compose start otel-collector | Out-Null
    } finally { Pop-Location }
    Start-Sleep -Seconds 20
}

function Assert-StackHealthy {
    Push-Location $DemoDir
    try {
        $rows = & docker @compose ps -a --format "{{.Name}}|{{.State}}|{{.Status}}"
    } finally { Pop-Location }
    $state = @{}
    foreach ($row in $rows) {
        $parts = "$row" -split "\|"
        if ($parts.Count -ge 3) { $state[$parts[0]] = "$($parts[1]) $($parts[2])" }
    }
    $problems = @()
    foreach ($name in $expected) {
        if (-not $state.ContainsKey($name)) { $problems += "$name : not present"; continue }
        $status = $state[$name]
        # `ps -a` also lists exited and paused containers, which `ps` alone would hide.
        if ($status -notlike "running *" -or $status -match "unhealthy") {
            $problems += "$name : $status"
        }
    }
    if ($problems) { throw "stack not healthy:`n$($problems -join "`n")" }
}

Write-Log "campaign start: $Chunks chunks x $PerChunk experiments, base seed $BaseSeed, out $OutRoot"
for ($c = 0; $c -lt $Chunks; $c++) {
    $seed = $BaseSeed + $c
    $last = Join-Path $OutRoot ("otel-{0}-{1:d4}" -f $seed, ($PerChunk - 1))
    if (Test-Path (Join-Path $last "manifest.json")) { Write-Log "chunk $c (seed $seed) already complete, skipping"; continue }
    Assert-StackHealthy
    # Whatever the previous chunk (or an interrupted run) left behind stops here.
    Reset-Stack
    Clear-Capture
    Write-Log "chunk $c (seed $seed) starting"
    $code = Invoke-Logged $rca @(
        "bench", "run-campaign", "--n", $PerChunk, "--seed", $seed,
        "--out-root", $OutRoot, "--capture-dir", $capture, "--flags-path", $flags,
        "--warmup-s", $WarmupS, "--fault-min-s", $FaultMinS, "--fault-max-s", $FaultMaxS,
        "--cooldown-s", $CooldownS, "--settle-s", $SettleS, "--offset-max-s", $OffsetMaxS)
    if ($code -ne 0) {
        Write-Log "chunk $c failed (exit $code); resetting and stopping"
        Reset-Stack
        exit $code
    }
    Write-Log "chunk $c done"
}
Reset-Stack
Write-Log "campaign complete"
