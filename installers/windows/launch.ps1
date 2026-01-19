# Transcribe Service Launcher for Windows
$ErrorActionPreference = "Stop"

# Get installation directory (script is in bin/ subdirectory, so go up one level)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$installDir = Split-Path -Parent $scriptDir
$binDir = Join-Path $installDir "bin"
$venvDir = Join-Path $installDir "venv"
$appDir = Join-Path $installDir "transcribe-service"
$modelsDir = Join-Path $installDir "models"
$dataDir = Join-Path $env:APPDATA "ivrit.ai\transcribe-service"
$logFile = Join-Path $dataDir "app.log"
$launchLogFile = Join-Path $dataDir "launch.log"
$pidFile = Join-Path $dataDir "app.pid"

# Add bin directory to PATH for ffmpeg
$env:PATH = "$binDir;$env:PATH"

# Ensure data directory exists
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

# Check if already running
if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile
    try {
        $process = Get-Process -Id $oldPid -ErrorAction Stop
        Write-Host "Transcribe service is already running (PID: $oldPid)"
        Write-Host "Opening browser..."
        Start-Sleep -Seconds 1
        Start-Process "http://localhost:4500"
        exit 0
    } catch {
        Remove-Item $pidFile -Force
    }
}

# Activate virtual environment and launch app
Write-Host "Starting transcribe service..."
$pythonExe = Join-Path $venvDir "Scripts\python.exe"

# Start the application in the background
$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = $pythonExe
$processInfo.Arguments = "app.py --local --data-dir `"$dataDir`" --models-dir `"$modelsDir`""
$processInfo.WorkingDirectory = $appDir
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $processInfo

# Redirect output to log files
$outputHandler = {
    param($sender, $e)
    if ($e.Data) {
        Add-Content -Path $logFile -Value $e.Data
    }
}

$errorHandler = {
    param($sender, $e)
    if ($e.Data) {
        Add-Content -Path $launchLogFile -Value $e.Data
    }
}

$process.add_OutputDataReceived($outputHandler)
$process.add_ErrorDataReceived($errorHandler)

$process.Start() | Out-Null
$process.BeginOutputReadLine()
$process.BeginErrorReadLine()

$appPid = $process.Id
$appPid | Out-File -FilePath $pidFile -Encoding ASCII

Write-Host "Transcribe service started (PID: $appPid)"
Write-Host "Log file: $logFile"
Write-Host "Error log file: $launchLogFile"

# Wait for server to start (poll for up to 20 seconds)
Write-Host "Waiting for server to start..."
$maxWait = 20
$elapsed = 0
$serverUp = $false

while ($elapsed -lt $maxWait) {
    # Check if process is still alive
    if ($process.HasExited) {
        Write-Host "ERROR: Server process exited unexpectedly" -ForegroundColor Red
        Write-Host "Check the error log for details: $launchLogFile" -ForegroundColor Red
        exit 1
    }
    
    # Try to connect to server using curl (faster and more reliable than Invoke-WebRequest)
    $null = curl.exe -s -o $null -w "%{http_code}" http://localhost:4500 2>$null
    if ($LASTEXITCODE -eq 0) {
        $serverUp = $true
        break
    }
    
    Start-Sleep -Seconds 1
    $elapsed++
}

if (-not $serverUp) {
    Write-Host "ERROR: Server failed to start after $maxWait seconds" -ForegroundColor Red
    Write-Host "Check the error log for details: $launchLogFile" -ForegroundColor Red
    Write-Host "Server process (PID: $appPid) may still be running" -ForegroundColor Yellow
    exit 1
}

Write-Host "Server is up after $elapsed seconds" -ForegroundColor Green

# Launch browser
Write-Host "Opening browser..."
Start-Process "http://localhost:4500"

Write-Host "Done! The service is running in the background." -ForegroundColor Green
Write-Host "To stop the service, run: Stop-Process -Id $appPid"
