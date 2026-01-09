# Transcribe Service Windows Installer
# Usage:
#   Install latest release from default repo:
#     iwr -useb https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/windows/install-windows.ps1 | iex
#
#   Install from any branch or tag (specify repo once):
#     $env:REPO_PATH="ivrit-ai/transcribe-service/main"; iwr -useb https://raw.githubusercontent.com/$env:REPO_PATH/installers/windows/install-windows.ps1 | iex
#     $env:REPO_PATH="ivrit-ai/transcribe-service/yairl/onprem"; iwr -useb https://raw.githubusercontent.com/$env:REPO_PATH/installers/windows/install-windows.ps1 | iex
#     $env:REPO_PATH="ivrit-ai/transcribe-service/v1.0.0"; iwr -useb https://raw.githubusercontent.com/$env:REPO_PATH/installers/windows/install-windows.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Transcribe Service Windows Installer" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Parse REPO_PATH environment variable
# Format: org/repo/ref-name (where ref-name can be a branch or tag)
# If not specified, defaults to latest release from ivrit-ai/transcribe-service
if ($env:REPO_PATH) {
    $pathParts = $env:REPO_PATH -split '/'
    
    if ($pathParts.Count -lt 3) {
        Write-Host "Error: Invalid REPO_PATH format" -ForegroundColor Red
        Write-Host "Expected: org/repo/ref-name (e.g., ivrit-ai/transcribe-service/main)" -ForegroundColor Red
        Write-Host "Got: $env:REPO_PATH" -ForegroundColor Red
        exit 1
    }
    
    $githubRepo = "$($pathParts[0])/$($pathParts[1])"
    $refName = $pathParts[2..($pathParts.Count - 1)] -join '/'
    $githubRef = $refName
    
    # Check if this is a tag or branch by querying GitHub API
    Write-Host "Checking if $githubRef is a tag or branch..."
    try {
        $tagResponse = Invoke-WebRequest -Uri "https://api.github.com/repos/$githubRepo/git/refs/tags/$githubRef" -Method Head -ErrorAction SilentlyContinue
        $isTag = $true
        Write-Host "Detected as tag: $githubRef"
    } catch {
        $isTag = $false
        Write-Host "Detected as branch: $githubRef"
    }
} else {
    # Default to latest release from ivrit-ai/transcribe-service
    $githubRepo = "ivrit-ai/transcribe-service"
    $githubRef = "latest"
    $isTag = $true
}

Write-Host "Using GitHub repository: $githubRepo"
Write-Host "Using reference: $githubRef"
Write-Host ""

# Check architecture - only x64 is supported
$arch = [System.Environment]::Is64BitOperatingSystem
if (-not $arch) {
    Write-Host "Error: This installer only supports 64-bit Windows." -ForegroundColor Red
    exit 1
}

# Configuration
$installDir = (Get-Location).Path
$uvDir = Join-Path $installDir "uv"
$binDir = Join-Path $installDir "bin"
$appDir = Join-Path $installDir "transcribe-service"
$venvDir = Join-Path $installDir "venv"
$modelsDir = Join-Path $installDir "models"
$dataDir = Join-Path $env:APPDATA "ivrit.ai\transcribe-service"
$modelUrl = "https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin"
$installLog = Join-Path $installDir "install.log"

# Start logging
Start-Transcript -Path $installLog -Append

# Check if installation already exists
$existingDirs = @()
if (Test-Path $appDir) { $existingDirs += "transcribe-service\" }
if (Test-Path $uvDir) { $existingDirs += "uv\" }
if (Test-Path $venvDir) { $existingDirs += "venv\" }
if (Test-Path $binDir) { $existingDirs += "bin\" }

if ($existingDirs.Count -gt 0) {
    Write-Host ""
    Write-Host "===================================" -ForegroundColor Yellow
    Write-Host "Existing Installation Detected" -ForegroundColor Yellow
    Write-Host "===================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "An installation already exists in this directory:"
    Write-Host "  $installDir"
    Write-Host ""
    foreach ($dir in $existingDirs) {
        Write-Host "  - $dir"
    }
    Write-Host ""
    $reply = Read-Host "Do you want to uninstall and reinstall? (y/N)"
    if ($reply -notmatch '^[Yy]$') {
        Write-Host "Installation aborted."
        Stop-Transcript
        exit 0
    }
    
    Write-Host "Removing existing installation..."
    if (Test-Path $appDir) { Remove-Item -Recurse -Force $appDir }
    if (Test-Path $uvDir) { Remove-Item -Recurse -Force $uvDir }
    if (Test-Path $binDir) { Remove-Item -Recurse -Force $binDir }
    if (Test-Path $venvDir) { Remove-Item -Recurse -Force $venvDir }
    $versionFile = Join-Path $installDir "VERSION"
    if (Test-Path $versionFile) { Remove-Item -Force $versionFile }
    Write-Host "✓ Existing installation removed" -ForegroundColor Green
    Write-Host ""
}

# Installation directory
Write-Host "Installing to: $installDir"

# Step 1: Download and install uv
Write-Host ""
Write-Host "Step 1/8: Downloading uv..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
$uvDownloadUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
Write-Host "Downloading from: $uvDownloadUrl"
$uvZip = Join-Path $uvDir "uv.zip"
Start-BitsTransfer -Source $uvDownloadUrl -Destination $uvZip -Description "Downloading uv"
Expand-Archive -Path $uvZip -DestinationPath $uvDir -Force
Remove-Item $uvZip
Write-Host "✓ uv installed successfully" -ForegroundColor Green

# Step 2: Download transcribe-service release
Write-Host ""
Write-Host "Step 2/8: Downloading transcribe-service..." -ForegroundColor Cyan
if ($githubRef -eq "latest") {
    # Get the latest release tag
    Write-Host "Fetching latest release..."
    $releaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$githubRepo/releases/latest"
    $releaseUrl = $releaseInfo.tarball_url
    $githubRef = $releaseInfo.tag_name
    if (-not $releaseUrl -or -not $githubRef) {
        Write-Host "Error: Could not fetch latest release" -ForegroundColor Red
        Stop-Transcript
        exit 1
    }
    Write-Host "Found latest release: $githubRef"
} elseif ($isTag) {
    # It's a tag
    Write-Host "Using tag: $githubRef"
    $releaseUrl = "https://api.github.com/repos/$githubRepo/tarball/refs/tags/$githubRef"
} else {
    # It's a branch
    Write-Host "Using branch: $githubRef"
    $releaseUrl = "https://api.github.com/repos/$githubRepo/tarball/$githubRef"
}

Write-Host "Downloading from GitHub (ref: $githubRef)..."
if (Test-Path $appDir) { Remove-Item -Recurse -Force $appDir }
New-Item -ItemType Directory -Force -Path $appDir | Out-Null

# Download tarball
$tarballPath = Join-Path $installDir "temp.tar.gz"
Start-BitsTransfer -Source $releaseUrl -Destination $tarballPath -Description "Downloading transcribe-service"

# Extract tarball (requires tar.exe which is available in Windows 10+ by default)
tar -xzf $tarballPath -C $appDir --strip-components=1
Remove-Item $tarballPath
Write-Host "✓ transcribe-service downloaded successfully" -ForegroundColor Green

# Fetch commit hash for the version
Write-Host "Fetching commit hash for $githubRef..."
try {
    if ($isTag) {
        # For tags, try to get the commit hash from the tag reference
        $refInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$githubRepo/git/refs/tags/$githubRef" -ErrorAction SilentlyContinue
        $commitHash = $refInfo.object.sha
        # If it's an annotated tag, we need to dereference it
        if (-not $commitHash) {
            $commitInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$githubRepo/commits/$githubRef"
            $commitHash = $commitInfo.sha
        }
    } else {
        # For branches, get the latest commit hash
        $commitInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$githubRepo/commits/$githubRef"
        $commitHash = $commitInfo.sha
    }
    
    if ($commitHash) {
        # Truncate to first 8 characters for brevity
        $commitHash = $commitHash.Substring(0, 8)
        Write-Host "✓ Commit hash: $commitHash" -ForegroundColor Green
    } else {
        $commitHash = "unknown"
        Write-Host "Warning: Could not fetch commit hash, using 'unknown'" -ForegroundColor Yellow
    }
} catch {
    $commitHash = "unknown"
    Write-Host "Warning: Could not fetch commit hash, using 'unknown'" -ForegroundColor Yellow
}

# Create VERSION file
$versionFile = Join-Path $installDir "VERSION"
if ($isTag) {
    $refType = "tag"
} else {
    $refType = "branch"
}
"${githubRepo}/${githubRef}@${commitHash} (${refType})" | Out-File -FilePath $versionFile -Encoding UTF8
Write-Host "✓ Version file created: $versionFile" -ForegroundColor Green
Write-Host "   ${githubRepo}/${githubRef}@${commitHash} (${refType})"

# Step 3: Create virtual environment with Python 3.13
Write-Host ""
Write-Host "Step 3/8: Creating virtual environment with Python 3.13..." -ForegroundColor Cyan
& "$uvDir\uv.exe" venv $venvDir --python 3.13
Write-Host "✓ Virtual environment created successfully" -ForegroundColor Green

# Step 4: Install requirements
Write-Host ""
Write-Host "Step 4/8: Installing requirements..." -ForegroundColor Cyan
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
& "$uvDir\uv.exe" pip install --python $pythonExe -r "$appDir\requirements.txt"
Write-Host "✓ Requirements installed successfully" -ForegroundColor Green

# Step 5: Install ivrit[all]
Write-Host ""
Write-Host "Step 5/8: Installing ivrit[all]..." -ForegroundColor Cyan
& "$uvDir\uv.exe" pip install --python $pythonExe "ivrit[all]"
Write-Host "✓ ivrit[all] installed successfully" -ForegroundColor Green

# Step 6: Create symlinks/copies for ffmpeg
Write-Host ""
Write-Host "Step 6/8: Setting up ffmpeg..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

# Get ffmpeg path from imageio-ffmpeg
$ffmpegPath = & $pythonExe -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if (-not $ffmpegPath -or -not (Test-Path $ffmpegPath)) {
    Write-Host "Error: Could not find ffmpeg binary from imageio-ffmpeg" -ForegroundColor Red
    Stop-Transcript
    exit 1
}

Write-Host "Found ffmpeg at: $ffmpegPath"

# Copy ffmpeg to bin directory (symlinks require admin privileges on Windows)
$ffmpegDest = Join-Path $binDir "ffmpeg.exe"
Copy-Item -Path $ffmpegPath -Destination $ffmpegDest -Force
Write-Host "✓ Copied ffmpeg to: $ffmpegDest" -ForegroundColor Green

# Step 7: Create data directory and download model
Write-Host ""
Write-Host "Step 7/8: Setting up data directory and downloading model..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
Write-Host "Data directory: $dataDir"
$modelFile = Join-Path $modelsDir "ivrit-ggml-model.bin"
if (Test-Path $modelFile) {
    Write-Host "Model already exists at $modelFile"
    $reply = Read-Host "Do you want to re-download it? (y/N)"
    if ($reply -match '^[Yy]$') {
        Write-Host "Downloading model (this may take a while)..."
        Start-BitsTransfer -Source $modelUrl -Destination $modelFile -Description "Downloading model" -DisplayName "ivrit-ggml-model.bin"
        Write-Host "✓ Model downloaded successfully" -ForegroundColor Green
    } else {
        Write-Host "Skipping model download"
    }
} else {
    Write-Host "Downloading model (this may take a while)..."
    Start-BitsTransfer -Source $modelUrl -Destination $modelFile -Description "Downloading model" -DisplayName "ivrit-ggml-model.bin"
    Write-Host "✓ Model downloaded successfully" -ForegroundColor Green
}

# Step 8: Create launcher scripts and desktop shortcut
Write-Host ""
Write-Host "Step 8/8: Creating launcher scripts..." -ForegroundColor Cyan

# Copy the launch.ps1 script from the installers directory
$launchScriptSource = Join-Path $appDir "installers\windows\launch.ps1"
$launchScriptDest = Join-Path $installDir "launch.ps1"
Copy-Item -Path $launchScriptSource -Destination $launchScriptDest -Force

# Create a batch file wrapper for easier launching
$batchFile = Join-Path $installDir "ivrit.ai.bat"
@"
@echo off
powershell.exe -ExecutionPolicy Bypass -File "$launchScriptDest"
"@ | Out-File -FilePath $batchFile -Encoding ASCII

# Create a desktop shortcut
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath("Desktop"), "ivrit.ai Transcribe.lnk"))
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launchScriptDest`""
$Shortcut.WorkingDirectory = $installDir
$Shortcut.IconLocation = Join-Path $appDir "static\favicon.ico"
$Shortcut.Description = "ivrit.ai Transcribe Service"
$Shortcut.Save()

Write-Host "✓ Launcher scripts created successfully" -ForegroundColor Green
Write-Host "✓ Desktop shortcut created" -ForegroundColor Green

# Installation complete
Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "Installation directory: $installDir"
Write-Host "Models directory: $modelsDir"
Write-Host "Data directory: $dataDir"
Write-Host "Launcher: $batchFile"
Write-Host "Desktop shortcut: Desktop\ivrit.ai Transcribe.lnk"
Write-Host "Installation log: $installLog"
Write-Host ""
Write-Host "To start the transcribe service:"
Write-Host "  - Double-click the 'ivrit.ai Transcribe' shortcut on your desktop, or"
Write-Host "  - Run: $batchFile"
Write-Host ""

Stop-Transcript
