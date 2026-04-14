# Shariah Stock Screener MVP

This is a beginner-friendly Streamlit app that checks one stock ticker at a time using a simple and transparent Shariah screening methodology.

Important: this is a **methodology-based prototype tool**, not a fatwa or religious ruling.

## What the app does

- Lets you enter one US stock ticker
- Fetches basic company and financial data from Yahoo Finance
- Fetches best-effort income-screen data from SEC EDGAR
- Runs a simple business activity screen
- Runs a best-effort income generation screen from SEC filing facts
- Runs simple financial ratio checks
- Shows a final result:
  - `Compliant`
  - `Non-compliant`
  - `Insufficient data`

## Important limitations

- This app uses **Yahoo Finance via `yfinance`** for prototype purposes.
- This app also uses **SEC EDGAR JSON APIs** for best-effort filing-based income checks.
- Yahoo Finance is convenient for beginners, but it is **not an official institutional data feed**.
- The SEC income screen only works when the company filing exposes usable XBRL facts.
- The parser now checks more SEC concept types, including interest income, dividend income, investment income, and some non-operating income concepts.
- If the SEC data does not expose a clean possible non-compliant income line item, the app will show `Insufficient data` instead of guessing.
- Some tickers may have missing, delayed, or incomplete fields.
- The business activity screen is only a **placeholder** based on sector and industry text.
- Detailed revenue-level screening is **not implemented yet**.
- This tool should not be treated as absolute religious certainty.

## Project files

- `app.py` = Streamlit user interface
- `data_fetcher.py` = gets stock data
- `sec_parser.py` = fetches best-effort SEC filing facts
- `screener.py` = runs the screening logic
- `methodology.py` = stores the screening rules and thresholds
- `utils.py` = helper functions
- `requirements.txt` = Python packages

## Local setup

### 1. Make sure Python is installed

You need Python 3.10 or newer.

To check:

```powershell
python --version
```

### 2. Open the project folder in your terminal

Example:

```powershell
cd "C:\Users\ali_m\OneDrive\Documents\New project"
```

### 3. Install the required packages

```powershell
pip install -r requirements.txt
```

### 3b. Recommended for SEC access

The SEC asks apps to send a declared user agent.

You can set one in PowerShell like this:

```powershell
$env:SEC_USER_AGENT="ShariahStockScreenerMVP your-email@example.com"
```

If you do not set this, the app uses a fallback value, but setting your own contact value is better and more aligned with SEC guidance.

### 4. Run the app

```powershell
streamlit run app.py
```

### 5. Open the local link

Streamlit will show a local web address in the terminal, usually:

```text
http://localhost:8501
```

Open that link in your browser.

## How to use the app

1. Enter a ticker like `AAPL`
2. Click `Run Screening`
3. Read the result sections:
   - methodology used
   - business activity screen
   - financial ratio screen
   - income generation screen
   - final verdict
   - limitations and disclaimer

## Example behavior

### Example 1: likely valid stock

If you enter `AAPL`, the app should:

- fetch Apple company data
- show the methodology used
- calculate available ratios
- try to find possible non-core income and revenue from recent SEC filing facts
- return a final verdict based on the configured rules

### Example 2: invalid ticker

If you enter `NOTAREALSTOCK`, the app should:

- not crash
- show a clear error message
- explain that no usable stock data was returned

### Example 3: partial data

If a ticker is missing important fields, the app should:

- still show any data it could fetch
- mark unavailable checks clearly
- return `Insufficient data` when needed

## Deploying on Streamlit Cloud

### 1. Put this project in a GitHub repository

Your repo should include:

- `app.py`
- `requirements.txt`
- the other `.py` files

### 2. Go to Streamlit Community Cloud

Open:

[https://share.streamlit.io/](https://share.streamlit.io/)

### 3. Connect your GitHub repo

Choose:

- your repository
- branch
- main file: `app.py`

### 4. Deploy

Streamlit Cloud will install the packages from `requirements.txt` and run the app.

## Notes about the current methodology

The MVP includes:

- a simple business activity screen using sector and industry keywords
- a best-effort SEC filing income screen using reported XBRL facts from a wider concept list
- financial thresholds stored in `methodology.py`

The thresholds are easy to edit later in one place.

## Next improvements after MVP

- Add better business activity screening with segment revenue data
- Add a more robust and scholarly configurable methodology
- Add support for more exchanges and regions
- Add a history of previous screenings
- Add downloadable reports
- Add unit tests
- Replace Yahoo Finance with a stronger data provider
