#requires -Version 7.0
param(
    [Parameter(Mandatory)][string]$PromptId,
    [Parameter(Mandatory)][int]$ComfyProcessId,
    [Parameter(Mandatory)][string]$LogPath,
    [Parameter(Mandatory)][string]$OutputPrefix,
    [string]$ServerUrl = 'http://127.0.0.1:7188',
    [ValidateRange(5, 60)][int]$IntervalSeconds = 20,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
if (-not ([uri]$ServerUrl).IsLoopback) { throw 'Use a loopback ComfyUI URL.' }
$OutputPrefix = [System.IO.Path]::GetFullPath($OutputPrefix)
$null = New-Item -ItemType Directory -Force -Path (Split-Path $OutputPrefix)
$telemetryPath = $OutputPrefix + '.jsonl'
if (Test-Path -LiteralPath $telemetryPath) { throw "Output already exists: $telemetryPath" }
$utf8 = New-Object System.Text.UTF8Encoding($false)
$totalMemory = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize * 1KB
$requestSaved = $false

do {
    $row = [ordered]@{ time = (Get-Date).ToString('o'); prompt_id = $PromptId; state = 'unknown'; errors = @() }
    $finished = $false
    try {
        $queue = Invoke-RestMethod -Uri ($ServerUrl.TrimEnd('/') + '/queue') -TimeoutSec 6
        $active = @($queue.queue_running | Where-Object { $_[1] -eq $PromptId })
        $pending = @($queue.queue_pending | Where-Object { $_[1] -eq $PromptId })
        $row.running = @($queue.queue_running).Count
        $row.pending = @($queue.queue_pending).Count
        if ($active.Count -gt 0) {
            $row.state = 'running'
            if (-not $requestSaved) {
                [System.IO.File]::WriteAllText($OutputPrefix + '.request.json',
                    (ConvertTo-Json -InputObject $active[0] -Depth 100), $utf8)
                $requestSaved = $true
            }
        } elseif ($pending.Count -gt 0) {
            $row.state = 'pending'
        } else {
            $history = Invoke-RestMethod -Uri ($ServerUrl.TrimEnd('/') + '/history/' + $PromptId) -TimeoutSec 6
            $entry = $history.PSObject.Properties[$PromptId]
            if ($null -ne $entry) {
                $result = $entry.Value
                $row.state = $result.status.status_str
                $row.completed = $result.status.completed
                $row.last_event = $result.status.messages[-1][0]
                [System.IO.File]::WriteAllText($OutputPrefix + '.history.json',
                    (ConvertTo-Json -InputObject $result -Depth 100), $utf8)
                $finished = $true
            }
        }
    } catch {
        $row.errors += 'ComfyUI: ' + $_.Exception.Message
    }

    try {
        $gpuLines = & nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
        $row.gpus = @($gpuLines | ForEach-Object {
            $v = $_ -split ','
            [ordered]@{ index = [int]$v[0]; total_mib = [double]$v[1]; used_mib = [double]$v[2];
                utilization = [double]$v[3]; temperature = [double]$v[4]; power_w = [double]$v[5] }
        })
    } catch {
        $row.errors += 'GPU: ' + $_.Exception.Message
    }

    try {
        $counters = (Get-Counter -Counter '\GPU Process Memory(*)\Shared Usage',
            '\GPU Process Memory(*)\Dedicated Usage', '\Processor(_Total)\% Processor Time',
            '\Memory\Available Bytes').CounterSamples
        $processCounters = @($counters | Where-Object { $_.InstanceName -like "pid_${ComfyProcessId}_*" })
        if ($processCounters.Count -gt 0) {
            $row.comfy_shared_gib = [math]::Round((($processCounters | Where-Object {
                $_.Path.EndsWith('\shared usage') } | Measure-Object CookedValue -Sum).Sum / 1GB), 3)
            $row.comfy_dedicated_gib = [math]::Round((($processCounters | Where-Object {
                $_.Path.EndsWith('\dedicated usage') } | Measure-Object CookedValue -Sum).Sum / 1GB), 3)
        }
        $available = ($counters | Where-Object { $_.Path.EndsWith('\available bytes') }).CookedValue
        $row.memory_used_gib = [math]::Round(($totalMemory - $available) / 1GB, 3)
        $row.memory_total_gib = [math]::Round($totalMemory / 1GB, 3)
        $row.cpu_percent = [math]::Round(($counters | Where-Object {
            $_.Path.EndsWith('\% processor time') }).CookedValue, 1)
    } catch {
        $row.errors += 'Windows counters: ' + $_.Exception.Message
    }

    try {
        if ($row.state -eq 'running' -or $row.running -eq 0) {
            $tail = @(Get-Content -LiteralPath $LogPath -Tail 100 -Encoding UTF8)
            $row.last_progress = @($tail | Where-Object {
                $_ -match 'PJ.*(采样|精修|预编码|准备完成)|Prompt executed|Processing interrupted|Exception|out of memory'
            } | Select-Object -Last 1)
        }
    } catch {
        $row.errors += 'Log: ' + $_.Exception.Message
    }
    $line = ConvertTo-Json -InputObject $row -Depth 8 -Compress
    [System.IO.File]::AppendAllText($telemetryPath, $line + [Environment]::NewLine, $utf8)
    Write-Output $line
    if ($Once -or $finished) { break }
    Start-Sleep -Seconds $IntervalSeconds
} while ($true)
