# ERA5 daily multi-location research dataset

Prepared data: `daily.npz`, shape **16,436 days × 7 locations × 6 variables** (115,052 location-days), 1981-01-01 through 2025-12-31. There are no missing dates or nonfinite values in this prepared benchmark.

Location order: Paris, London, Madrid, Rome, Athens, Berlin, Warsaw. The latitude/longitude arrays identify requested representative locations; the exact grid coordinates returned by the land-cell query are recorded in `manifest.json` for every response. They are not station coordinates. Elevation downscaling is disabled.

Variable order and units:

1. `temperature_2m_max`: °C, daily maximum.
2. `temperature_2m_min`: °C, daily minimum.
3. `pressure_msl_mean`: hPa, daily mean.
4. `soil_moisture_0_to_7cm_mean`: m³/m³, daily mean.
5. `relative_humidity_2m_mean`: %, daily mean.
6. `precipitation_sum`: mm, daily total.

Attribution: ERA5, Copernicus Climate Change Service / ECMWF; daily data served by Open-Meteo, [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), [CC BY 4.0 and attribution information](https://open-meteo.com/en/terms). Modifications: extraction at requested land locations, six-variable selection, UTC daily aggregation by the provider, response concatenation and conversion to NumPy float32. Raw JSON responses preserve the original returned values and units.

The manifest includes requests, response coordinates/units, per-response retrieval timestamps and SHA-256 hashes. Raw responses and the prepared NPZ are retained locally and excluded from Git; `python -m heatcast.download_sites` reproduces them subject to the provider's quota. API credits are weighted by time span, so rate-limit retries may be necessary.

A partially downloaded Stockholm response is cached separately from the seven-location manifest. Stockholm is excluded from the prepared benchmark due to incomplete download, not model performance. Expanding coverage requires completing a separate dataset and rerunning its full evaluation. No Scandinavian forecast claim is made from this benchmark.
