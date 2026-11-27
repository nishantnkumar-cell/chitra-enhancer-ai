<#
.SYNOPSIS
  Installs Chitra Enhancer as a Windows service for your Tailscale network.
.DESCRIPTION
  Run from an administrator PowerShell. The service starts with Windows, whether
  or not anyone is signed in, runs under its own restricted account
  (NT SERVICE\ChitraEnhancer), listens only on this computer's Tailscale
  address, and asks for a secret key with every request.

  Program: C:\Program Files\Chitra Enhancer Service
  Models, settings, key, log and work files: C:\ProgramData\ChitraEnhancer
  (readable by administrators and the service only)

  Running it again updates the program and keeps the key.
.PARAMETER Program
  A folder with chitra-enhance.exe, the ONNX Runtime DLLs, ffmpeg.exe and
  ffprobe.exe. Default: the folder this script is in.
.PARAMETER Models
  A folder of .onnx models to give the service (for example the folder
  `chitra-enhance models list` shows).
.PARAMETER Model
  The model used when a sender asks for none, by file name without .onnx.
.PARAMETER Port
  The port to listen on. Default 8765.
#>
param(
  [string]$Program = $PSScriptRoot,
  [Parameter(Mandatory = $true)][string]$Models,
  [string]$Model = "realesr-general-x4v3_fp16",
  [ValidateRange(1024, 65535)][int]$Port = 8765
)
$ErrorActionPreference = "Stop"
$name    = "ChitraEnhancer"
$account = "NT SERVICE\$name"
$rule    = "Chitra Enhancer (Tailscale only)"
$target  = Join-Path $env:ProgramFiles "Chitra Enhancer Service"
$state   = Join-Path $env:ProgramData "ChitraEnhancer"
$config  = Join-Path $state "config.json"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this from an administrator PowerShell." }
foreach ($f in "chitra-enhance.exe", "onnxruntime.dll", "ffmpeg.exe", "ffprobe.exe") {
  if (-not (Test-Path (Join-Path $Program $f))) { throw "$f is not in $Program" }
}
if (-not (Test-Path (Join-Path $Models "$Model.onnx"))) { throw "$Model.onnx is not in $Models" }

$tailscale = Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"
if (-not (Test-Path $tailscale)) { throw "Tailscale is not installed." }
$address = (& $tailscale ip -4 | Select-Object -First 1).Trim()
if ($address -notmatch '^100\.') { throw "Tailscale has no address; sign in to it first." }

# Stop an earlier installation before replacing its files.
if (Get-Service $name -ErrorAction SilentlyContinue) {
  Stop-Service $name -Force -ErrorAction SilentlyContinue
  (Get-Service $name).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(60))
}

New-Item -ItemType Directory -Force $target, $state, "$state\models", "$state\jobs" | Out-Null
Copy-Item (Join-Path $Program "chitra-enhance.exe"), (Join-Path $Program "ffmpeg.exe"), (Join-Path $Program "ffprobe.exe") $target -Force
Copy-Item (Join-Path $Program "*.dll") $target -Force
foreach ($extra in "LICENSE", "THIRD_PARTY_LICENCES.txt", "MODEL_LICENCES.md", "USER_GUIDE.md") {
  if (Test-Path (Join-Path $Program $extra)) { Copy-Item (Join-Path $Program $extra) $target -Force }
}
Copy-Item (Join-Path $Models "*.onnx") "$state\models" -Force
$exe = Join-Path $target "chitra-enhance.exe"

# Keep the key of an earlier installation.
$key = $null
if (Test-Path $config) { $key = (Get-Content $config -Raw | ConvertFrom-Json).key }
if (-not $key) { $key = (& $exe new-key).Trim() }

if (-not (Get-Service $name -ErrorAction SilentlyContinue)) {
  New-Service -Name $name -DisplayName "Chitra Enhancer" -BinaryPathName "`"$exe`" serve --windows-service --config `"$config`"" -StartupType Automatic | Out-Null
}
function Invoke-Sc { & "$env:SystemRoot\System32\sc.exe" @args | Out-Null; if ($LASTEXITCODE -ne 0) { throw "sc.exe $($args[0]) failed ($LASTEXITCODE)" } }
Invoke-Sc config $name "obj=" $account "start=" "delayed-auto" "depend=" "Tailscale"
Invoke-Sc sidtype $name unrestricted
Invoke-Sc failure $name "reset=" "86400" "actions=" "restart/5000/restart/30000/restart/60000"
Invoke-Sc failureflag $name 1
Invoke-Sc description $name "Enhances pictures and short clips for your own computers over Tailscale."

# Settings, key and work files: administrators, Windows and the service only.
# (Set by well-known identifiers, so this works in every language.)
function Invoke-Acl { & "$env:SystemRoot\System32\icacls.exe" @args | Out-Null; if ($LASTEXITCODE -ne 0) { throw "icacls failed on $($args[0])" } }
Invoke-Acl $state /inheritance:r /grant:r "*S-1-5-32-544:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" "${account}:(OI)(CI)RX"
Invoke-Acl "$state\jobs" /grant:r "${account}:(OI)(CI)M"
if (-not (Test-Path "$state\service.log")) { New-Item -ItemType File "$state\service.log" | Out-Null }
Invoke-Acl "$state\service.log" /grant:r "${account}:M"

$settings = [ordered]@{
  listen  = @("${address}:$Port")
  key     = $key
  data    = "$state\jobs"
  models  = "$state\models"
  model   = $Model
  ffmpeg  = "$target\ffmpeg.exe"
  runtime = "$target\onnxruntime.dll"
  log     = "$state\service.log"
}
# UTF-8; the program accepts the byte order mark Windows adds.
$settings | ConvertTo-Json | Set-Content $config -Encoding UTF8

Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -LocalAddress $address -RemoteAddress "100.64.0.0/10" -Program $exe | Out-Null

Start-Service $name
Start-Sleep -Seconds 3
"Service: $((Get-Service $name).Status), running as $account"
"Address: http://${address}:$Port/  (open it in a browser on any of your Tailscale devices)"
"Key:     the `"key`" line in $config (open it as administrator)"
