<#
Stage 3 runner: trains LOCUS for every (anchors, seed) pair and logs each run.

  .\repro\run_stage3.ps1                          # full grid, in priority order
  .\repro\run_stage3.ps1 -Anchors 1024,512,256 -Seeds 42
  .\repro\run_stage3.ps1 -DryRun                  # print the plan only

- Finished runs (training_summary.json present) are skipped, so the script can be
  stopped with Ctrl+C and started again; an interrupted run is redone from scratch.
- Each run writes runs\main\a<A>_s<S>\ and runs\main\a<A>_s<S>.log, and appends a
  row to runs\registry.csv.
- Keeps Windows from sleeping while it runs.
#>
param(
    [int[]]$Anchors = @(1024, 512, 256, 128, 768, 2048),
    [int[]]$Seeds = @(42, 43, 44),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Code = Join-Path $Root "locus_code_release"
$OutDir = Join-Path $Root "runs\main"
$Registry = Join-Path $Root "runs\registry.csv"

# Micro-batches per batch of 112 models (measured on the RTX 3050 4 GB, see PLAN.md Stage 3)
$AccumSteps = @{ 128 = 2; 256 = 2; 512 = 2; 768 = 2; 1024 = 2; 2048 = 8 }

# Environment (same as repro\env.ps1, without needing activation)
$env:PATH = "$Root\.venv\Scripts;$env:PATH"
$env:VIRTUAL_ENV = "$Root\.venv"
$env:PYTHONUTF8 = "1"
$env:USE_TF = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Priority order: seed 42 for every budget first, then the other seeds
$Jobs = @()
foreach ($S in $Seeds) { foreach ($A in $Anchors) { $Jobs += [pscustomobject]@{ A = $A; S = $S } } }
$Jobs = @($Jobs | Sort-Object @{ Expression = { if ($_.S -eq 42) { 0 } else { 1 } } }, @{ Expression = { $_.S } }, @{ Expression = { [array]::IndexOf($Anchors, $_.A) } })

New-Item -ItemType Directory -Force $OutDir | Out-Null
if (-not (Test-Path $Registry)) {
    "run,anchors,seed,grad_accum_steps,start,end,minutes,exit_code,best_epoch,stopped_epoch,best_val_hit1,best_test_hit1,test_correctness_acc,git_commit,git_dirty" |
        Out-File -Encoding ascii $Registry
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

$commit = (git -C $Code rev-parse --short HEAD).Trim()
$dirty = if (git -C $Code status --porcelain -- scripts) { "yes" } else { "no" }

Write-Host "Stage 3: $($Jobs.Count) runs | code $commit (uncommitted changes: $dirty)"
Push-Location $Code
try {
    if (-not $DryRun) { [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) }
    $i = 0
    foreach ($job in $Jobs) {
        $i++
        $name = "a$($job.A)_s$($job.S)"
        $runDir = Join-Path $OutDir $name
        $k = $AccumSteps[$job.A]; if (-not $k) { $k = 2 }
        if (Test-Path (Join-Path $runDir "training_summary.json")) {
            Write-Host "[$i/$($Jobs.Count)] $name  done, skipping"
            continue
        }
        $argsLine = "--anchors-per-model $($job.A) --rng-seed $($job.S) --grad-accum-steps $k --models-dir ..\runs\main\$name"
        if ($DryRun) { Write-Host "[$i/$($Jobs.Count)] $name  python scripts\training_script.py $argsLine"; continue }

        $start = Get-Date
        Write-Host "[$i/$($Jobs.Count)] $name  started $($start.ToString('HH:mm'))  (K=$k)"
        cmd /c "python scripts\training_script.py $argsLine > ..\runs\main\$name.log 2>&1"
        $code = $LASTEXITCODE
        $end = Get-Date
        $minutes = [math]::Round(($end - $start).TotalMinutes, 1)

        $best = ""; $stopped = ""; $val1 = ""; $test1 = ""; $acc = ""
        $summaryPath = Join-Path $runDir "training_summary.json"
        if (Test-Path $summaryPath) {
            $s = Get-Content $summaryPath -Raw | ConvertFrom-Json
            $best = $s.best_epoch; $stopped = $s.stopped_epoch
            $val1 = [math]::Round($s.'best_val_hit@1', 4); $test1 = [math]::Round($s.'best_test_hit@1', 4)
            $acc = [math]::Round($s.best_eval.test.overall.correctness_prediction.correctness_pred_acc, 4)
            Write-Host ("    finished in {0} min | best epoch {1} | test hit@1 {2} | test acc {3}" -f $minutes, $best, $test1, $acc)
        } else {
            Write-Host "    FAILED (exit code $code) after $minutes min, see runs\main\$name.log" -ForegroundColor Red
        }
        "$name,$($job.A),$($job.S),$k,$($start.ToString('s')),$($end.ToString('s')),$minutes,$code,$best,$stopped,$val1,$test1,$acc,$commit,$dirty" |
            Out-File -Encoding ascii -Append $Registry
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Pop-Location
}
