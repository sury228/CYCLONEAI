$path = Join-Path $PSScriptRoot 'project_architecture.png'

Add-Type -AssemblyName System.Drawing
$img = New-Object System.Drawing.Bitmap(1200, 900)
$g = [System.Drawing.Graphics]::FromImage($img)
$g.Clear([System.Drawing.Color]::White)

$blue = [System.Drawing.Color]::FromArgb(234,242,255)
$edge = [System.Drawing.Color]::FromArgb(46,91,186)
$textColor = [System.Drawing.Color]::Black
$lineColor = [System.Drawing.Color]::FromArgb(62,74,89)

$titleFont = New-Object System.Drawing.Font('Arial', 20, [System.Drawing.FontStyle]::Bold)
$boxFont = New-Object System.Drawing.Font('Arial', 12, [System.Drawing.FontStyle]::Regular)
$blueBrush = New-Object System.Drawing.SolidBrush($blue)
$textBrush = New-Object System.Drawing.SolidBrush($textColor)
$edgePen = New-Object System.Drawing.Pen($edge, 2)
$linePen = New-Object System.Drawing.Pen($lineColor, 2)

$g.DrawString('CycloneVision AI Project Architecture', $titleFont, $textBrush, 250, 20)

$boxes = @(
    @{ x = 60; y = 120; w = 220; h = 140; label = 'Satellite Data`nCyclone_Images.h5' },
    @{ x = 340; y = 120; w = 220; h = 140; label = 'IBTrACS`nStorm Tracks' },
    @{ x = 620; y = 120; w = 220; h = 140; label = 'NWP / ERA5`nWeather Grids' },
    @{ x = 60; y = 340; w = 220; h = 150; label = 'process_cyclone_images.py`nLoad + normalize' },
    @{ x = 340; y = 340; w = 220; h = 150; label = 'process_ibtracs.py`nClean + categorize' },
    @{ x = 620; y = 340; w = 220; h = 150; label = 'fetch_nwp_data.py`nFetch / generate' },
    @{ x = 300; y = 580; w = 360; h = 170; label = 'data_fusion_pipeline.py`nFuse image + NWP + track data' },
    @{ x = 300; y = 820; w = 360; h = 110; label = 'train_val_test_split.py`nTrain / Val / Test split' },
    @{ x = 40; y = 900; w = 260; h = 100; label = 'Model 1`nVortex eye detection`n& segmentation' },
    @{ x = 330; y = 900; w = 260; h = 100; label = 'Model 2`nTrack forecasting' },
    @{ x = 620; y = 900; w = 260; h = 100; label = 'predict_track.py`nInference + map output' }
)

foreach ($b in $boxes) {
    $rect = New-Object System.Drawing.Rectangle($b.x, $b.y, $b.w, $b.h)
    $g.FillRectangle($blueBrush, $rect)
    $g.DrawRectangle($edgePen, $rect)
    $g.DrawString($b.label, $boxFont, $textBrush, ($b.x + 12), ($b.y + 18))
}

# arrows
$lines = @(
    @([System.Drawing.Point]::new(170, 260), [System.Drawing.Point]::new(170, 340)),
    @([System.Drawing.Point]::new(450, 260), [System.Drawing.Point]::new(450, 340)),
    @([System.Drawing.Point]::new(730, 260), [System.Drawing.Point]::new(730, 340)),
    @([System.Drawing.Point]::new(170, 490), [System.Drawing.Point]::new(400, 580)),
    @([System.Drawing.Point]::new(450, 490), [System.Drawing.Point]::new(500, 580)),
    @([System.Drawing.Point]::new(730, 490), [System.Drawing.Point]::new(500, 580)),
    @([System.Drawing.Point]::new(480, 750), [System.Drawing.Point]::new(480, 820)),
    @([System.Drawing.Point]::new(170, 1000), [System.Drawing.Point]::new(170, 1000)),
    @([System.Drawing.Point]::new(460, 1000), [System.Drawing.Point]::new(460, 1000)),
    @([System.Drawing.Point]::new(750, 1000), [System.Drawing.Point]::new(750, 1000))
)

foreach ($line in $lines) {
    $g.DrawLine($linePen, $line[0], $line[1])
}

$img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$img.Dispose()

Write-Host "Created architecture PNG: $path"
