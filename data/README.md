# Data

Raw data are not redistributed in this repository.

The analysis expects two inputs.

## 1. Malaria surveillance data

A cleaned record-level CSV containing at minimum:

- `year`
- `epidermiological week`

Each row is treated as one reported malaria surveillance record/case. The original project also contained fields such as disease/diagnostic code, age, sex, location and facility code, but they are not required for the weekly count model once the intended geographic subset has been created.

Suggested filename:

`peru_malaria_cleaned.csv`

## 2. Hourly weather data

An Excel file containing:

- `time`
- `temperature_2m (°C)`
- `rain (mm)`

Suggested filename:

`open_meteo_hourly.xlsx`

The script converts hourly timestamps to Peru local time using the same UTC-5 assumption used in the original analysis, reconstructs Sunday-Saturday epidemiological weeks, and aggregates:

- temperature -> weekly mean;
- rainfall -> weekly sum.

