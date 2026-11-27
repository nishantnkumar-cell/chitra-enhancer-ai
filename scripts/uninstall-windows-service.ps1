<#
.SYNOPSIS
  Removes the Chitra Enhancer Windows service.
.PARAMETER RemoveSettings
  Also delete C:\ProgramData\ChitraEnhancer (models, key and log).
#>
param([switch]$RemoveSettings)
$ErrorActionPreference = "Stop"
$name = "ChitraEnhancer"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this from an administrator PowerShell." }
if (Get-Service $name -ErrorAction SilentlyContinue) {
  Stop-Service $name -Force -ErrorAction SilentlyContinue
  (Get-Service $name).WaitForStatus("Stopped", [TimeSpan]::FromSeconds(60))
  & "$env:SystemRoot\System32\sc.exe" delete $name | Out-Null
}
Get-NetFirewallRule -DisplayName "Chitra Enhancer (Tailscale only)" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
$program = Join-Path $env:ProgramFiles "Chitra Enhancer Service"
if (Test-Path $program) { Remove-Item $program -Recurse -Force }
if ($RemoveSettings) {
  $state = Join-Path $env:ProgramData "ChitraEnhancer"
  if (Test-Path $state) { Remove-Item $state -Recurse -Force }
}
"Removed."
