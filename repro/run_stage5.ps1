<#
Stage 5 runner: ablations of the sentence encoder (paper Tables 7-8) and the embedding dimension
(Tables 9-10), anchors 1024 / 512 / 256, seed 42. mpnet and d=128 are reused from Stage 3.

  .\repro\run_stage5.ps1                     # all 24 runs, 1024-anchor runs first
  .\repro\run_stage5.ps1 -WaitPid 17340      # start after that process (e.g. the Stage 7 runner) exits (one GPU)
  .\repro\run_stage5.ps1 -Anchors 1024 -DryRun

- Each run writes runs\ablation\<tag>_a<A>_s42\ and its .log, and appends a row to runs\registry.csv.
- K (micro-batches of the 112 models) grows with anchors x z_dim so activations stay within 4 GB;
  for <= 1024 anchors K does not change the result, only the speed.
- Finished runs are skipped, so the script can be stopped and started again.
#>
param(
    [int[]]$Anchors = @(1024, 512, 256),
    [int]$WaitPid = 0,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Code = Join-Path $Root "locus_code_release"
$OutDir = Join-Path $Root "runs\ablation"
$Registry = Join-Path $Root "runs\registry.csv"
$Seed = 42

$env:PATH = "$Root\.venv\Scripts;$env:PATH"
$env:VIRTUAL_ENV = "$Root\.venv"
$env:PYTHONUTF8 = "1"
$env:USE_TF = "0"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$Variants = @(
    [pscustomobject]@{ Tag = "enc_albert";        Args = "--emb-col emb_paraphrase_albert_small_v2"; Z = 128 }
    [pscustomobject]@{ Tag = "enc_minilm";        Args = "--emb-col emb_all_minilm_l6_v2";           Z = 128 }
    [pscustomobject]@{ Tag = "enc_distilroberta"; Args = "--emb-col emb_all_distilroberta_v1";       Z = 128 }
)
foreach ($z in 16, 32, 64, 256, 512) { $Variants += [pscustomobject]@{ Tag = "z$z"; Args = "--z-dim $z"; Z = $z } }

$Jobs = foreach ($A in $Anchors) { foreach ($v in $Variants) { [pscustomobject]@{ A = $A; V = $v } } }

if ($WaitPid -and -not $DryRun) {
    Write-Host "waiting for process $WaitPid to exit"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 60 }
    Write-Host "process $WaitPid exited at $((Get-Date).ToString('HH:mm')), starting Stage 5"
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"; $ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

$commit = (git -C $Code rev-parse --short HEAD).Trim()
$dirty = if (git -C $Code status --porcelain -- scripts) { "yes" } else { "no" }
New-Item -ItemType Directory -Force $OutDir | Out-Null

Write-Host "Stage 5: $(@($Jobs).Count) runs | code $commit (uncommitted changes: $dirty)"
Push-Location $Code
try {
    if (-not $DryRun) { [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) }
    $i = 0
    foreach ($job in $Jobs) {
        $i++
        $name = "$($job.V.Tag)_a$($job.A)_s$Seed"
        $runDir = Join-Path $OutDir $name
        $k = [math]::Max(2, [int](2 * ($job.A / 1024) * ($job.V.Z / 128)))
        if (Test-Path (Join-Path $runDir "training_summary.json")) { Write-Host "[$i] $name  done, skipping"; continue }
        $argsLine = "--anchors-per-model $($job.A) --rng-seed $Seed --grad-accum-steps $k $($job.V.Args) --models-dir ..\runs\ablation\$name"
        if ($DryRun) { Write-Host "[$i] $name  python scripts\training_script.py $argsLine"; continue }

        $start = Get-Date
        Write-Host "[$i/$(@($Jobs).Count)] $name  started $($start.ToString('HH:mm'))  (K=$k)"
        cmd /c "python scripts\training_script.py $argsLine > ..\runs\ablation\$name.log 2>&1"
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
            Write-Host "    FAILED (exit code $code) after $minutes min, see runs\ablation\$name.log" -ForegroundColor Red
        }
        "abl_$name,$($job.A),$Seed,$k,$($start.ToString('s')),$($end.ToString('s')),$minutes,$code,$best,$stopped,$val1,$test1,$acc,$commit,$dirty" |
            Out-File -Encoding ascii -Append $Registry
    }
}
finally {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    Pop-Location
}
