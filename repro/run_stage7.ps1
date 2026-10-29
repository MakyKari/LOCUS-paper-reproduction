<#
Stage 7 runner: trains LOCUS on nested pools of 8..96 models (out of models 0-95) for the
new-model onboarding experiment (paper Fig. 3). Models 96-111 are held out.

  .\repro\run_stage7.ps1                    # pools 8, 16, 32, 64, 96
  .\repro\run_stage7.ps1 -Pools 8 -DryRun   # print the plan only

- Pool k = the first k models of one fixed permutation of 0-95 (numpy default_rng(0)), so pools are nested.
- The same index list goes to --model-idx-train/val/test (compute_val_metrics needs matching shapes).
- 1024 anchors, seed 42, K = 2 (exact for <= 1024 anchors). Finished runs are skipped.
- Each run writes runs\onboard\m<k>_s42\ and its .log, and appends a row to runs\registry.csv.
#>
param(
    [int[]]$Pools = @(8, 16, 32, 64, 96),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Code = Join-Path $Root "locus_code_release"
$OutDir = Join-Path $Root "runs\onboard"
$Registry = Join-Path $Root "runs\registry.csv"
$Anchors = 1024; $Seed = 42; $K = 2

$env:PATH = "$Root\.venv\Scripts;$env:PATH"
$env:VIRTUAL_ENV = "$Root\.venv"
$env:PYTHONUTF8 = "1"
$env:USE_TF = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$perm = (python -c "import numpy as np; print(','.join(map(str, np.random.default_rng(0).permutation(96))))").Trim().Split(",")

Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

$commit = (git -C $Code rev-parse --short HEAD).Trim()
$dirty = if (git -C $Code status --porcelain -- scripts) { "yes" } else { "no" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

Write-Host "Stage 7: pools $($Pools -join ', ') | code $commit (uncommitted changes: $dirty)"
Push-Location $Code
try {
    if (-not $DryRun) { [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) }
    foreach ($m in $Pools) {
        $name = "m$($m)_s$Seed"
        $runDir = Join-Path $OutDir $name
        if (Test-Path (Join-Path $runDir "training_summary.json")) { Write-Host "$name  done, skipping"; continue }
        $idx = ($perm[0..($m - 1)] | ForEach-Object { [int]$_ } | Sort-Object) -join ","
        $argsLine = "--anchors-per-model $Anchors --rng-seed $Seed --grad-accum-steps $K " +
                    "--model-idx-train $idx --model-idx-val $idx --model-idx-test $idx --models-dir ..\runs\onboard\$name"
        if ($DryRun) { Write-Host "$name  python scripts\training_script.py $argsLine"; continue }

        $start = Get-Date
        Write-Host "$name  started $($start.ToString('HH:mm'))"
        cmd /c "python scripts\training_script.py $argsLine > ..\runs\onboard\$name.log 2>&1"
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
            Write-Host ("    finished in {0} min | best epoch {1} | test acc {2}" -f $minutes, $best, $acc)
        } else {
            Write-Host "    FAILED (exit code $code) after $minutes min, see runs\onboard\$name.log" -ForegroundColor Red
        }
        "onboard_$name,$Anchors,$Seed,$K,$($start.ToString('s')),$($end.ToString('s')),$minutes,$code,$best,$stopped,$val1,$test1,$acc,$commit,$dirty" |
            Out-File -Encoding ascii -Append $Registry
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Pop-Location
}
