param(
    [string]$Game = 'E:\SteamLibrary\steamapps\common\SlayTheSpire',
    [string]$Workshop = (Join-Path (Split-Path (Split-Path $Game)) 'workshop/content/646570')
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $jars = @((Join-Path $Game 'desktop-1.0.jar'),
              (Join-Path $Workshop '1605060445/ModTheSpire.jar'),
              (Join-Path $Workshop '1605833019/BaseMod.jar'))
    foreach ($file in $jars) {
        if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing dependency: $file" }
    }
    $classes = Join-Path 'build' ('classes-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $classes | Out-Null
    $sources = @(Get-ChildItem -LiteralPath src -Filter '*.java' | ForEach-Object FullName)
    javac --release 8 -encoding UTF-8 -cp ($jars -join ';') -d $classes $sources
    if ($LASTEXITCODE) { throw 'Compilation failed' }
    jar cf build/SpireCombatAI.jar ModTheSpire.json -C $classes .
    if ($LASTEXITCODE) { throw 'Packaging failed' }
    Get-Item build/SpireCombatAI.jar
} finally { Pop-Location }
