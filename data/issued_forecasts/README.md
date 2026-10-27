# Explicit archived GFS runs for warning evaluation

This dataset contains 150 requested GFS Global initializations at 00 UTC, covering the exact runs needed by the predeclared development/test targets and 1-, 3-, and 7-day warning leads. Dates span April 23–September 28, 2026, with the unused June gap omitted by the request generator. Each run requests 12 days of hourly 2m temperature at the seven existing locations, using the actual ERA5 reference grid coordinates as query points, nearest grid selection, UTC, and disabled elevation downscaling.

149 runs contain all requested hours. The June 11 run returns missing temperature values and is preserved as an incomplete archive run. No later run fills it. The matched evaluation excludes development targets June 13, June 15 and June 19 across all tasks/leads. All 85 nominal July 8–September 30 test dates are complete. An archive gap does not prove a failure of the originating operational weather forecast.

`daily_runs.npz` contains:

- `initializations`: exact requested initialization datetimes.
- `valid_dates`: 12 UTC forecast dates per initialization.
- `forecast_tmax`: daily maxima from complete 24-hour hourly forecasts, shape (150, 12, 7); the missing run is NaN.
- `verification_dates`, `verification_tmax`: separate retrospective ERA5 verification from April 1–October 2, 2026.
- `sites`, `latitude`, `longitude`: site order and requested ERA5 verification grid coordinates. Actual GFS grid coordinates are preserved per response in the manifest.

`manifest.json` records exact request parameters, source/response hashes, retrieval timestamps, returned units and coordinates, the protocol hash, and each run's completeness status. Cached raw responses and request metadata are under `runs/`; verification is in `verification_2026.json`. Raw files and prepared arrays remain local and are ignored by Git. Run `python -m heatcast.download_issued` to reproduce or resume acquisition under the provider's access limits.

Forecast attribution: NOAA/NCEP GFS; archived run extraction by Open-Meteo. Verification and historical reference: Copernicus Climate Change Service/ECMWF ERA5, served by Open-Meteo. Data are used with CC BY 4.0 attribution. Transformations: selected locations, fixed model selection, complete UTC hourly-to-daily maximum aggregation, and NumPy conversion.

Sources: [Single Runs API](https://open-meteo.com/en/docs/single-runs-api), [GFS API](https://open-meteo.com/en/docs/gfs-api), [ERA5 archive](https://open-meteo.com/en/docs/historical-weather-api), [terms and attribution](https://open-meteo.com/en/terms).

This is a retrospective archive replay. Run initialization is not historical publication/receipt time. The evaluation assumes publication within 12 hours and waits until initialization +24 hours to issue a warning. Verification is unavailable to the inference function, and published-data delays are respected by the ERA5 persistence baseline. The timing policy and its limits are in `ISSUED_FORECAST_PROTOCOL.md` at the project root.
