<#
Weekly Chrono24 snapshot workflow for watchlists\budget_under_1000.txt.

Chrono24 blocks automated fetching (see watchlab/sources/chrono24.py), so
this cannot download pages itself -- saving them is a manual, human step.
This script handles the repeatable part around that: the dated folder, and
running calibrate + ingest once the pages are saved.

Usage, from the repo root, in your own terminal (not run automatically):

  1. .\scripts\ingest-chrono24.ps1
     Creates today's folder and prints the search URLs to save by hand.

  2. .\scripts\ingest-chrono24.ps1 -Ingest
     Once the .html files are saved into that folder, calibrates the parser
     against them and ingests whatever it found for today's date.

Repeat weekly (or every couple of weeks) so watchlab.sqlite3 accumulates
enough periods for `python -m watchlab report` to say something.
#>
param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [switch]$Ingest
)

$folder = "data\chrono24\$Date"

if (-not $Ingest) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Write-Host "Folder ready: $folder" -ForegroundColor Green
    Write-Host "For each line below: open the URL, Ctrl+S, save as 'Webpage, HTML only'," -ForegroundColor Cyan
    Write-Host "into that folder, named after the reference (e.g. SRPD55K1.html).`n" -ForegroundColor Cyan
    Get-Content "watchlists\budget_under_1000_search_urls.txt"
    Write-Host "`nWhen done saving: .\scripts\ingest-chrono24.ps1 -Ingest"
    exit 0
}

$files = Get-ChildItem "$folder\*.html" -ErrorAction SilentlyContinue
if (-not $files) {
    Write-Host "No .html files in $folder yet -- run this script without -Ingest first, save the pages, then re-run with -Ingest." -ForegroundColor Yellow
    exit 1
}

Write-Host "Calibrating against $($files.Count) saved page(s)..." -ForegroundColor Cyan
python -m watchlab calibrate $files.FullName

Write-Host "`nIf every page above shows 0 listings for both strategies, the selectors" -ForegroundColor Yellow
Write-Host "in watchlab/sources/chrono24.py need fixing against this real page --" -ForegroundColor Yellow
Write-Host "stop here and flag it rather than ingesting nothing useful." -ForegroundColor Yellow
Write-Host "`nIngesting for $Date ..." -ForegroundColor Cyan
python -m watchlab ingest $files.FullName --date $Date
