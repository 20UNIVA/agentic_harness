"""Example toolkit #1: plain functions (stateless). Registered as toolkit "finance".

    agentic-harness run --toolkits examples/tools/finance_tools.py ...
"""
from agentic_harness.tools import ToolError, tool

TOOLKIT_NAME = "finance"

# Fixed demo rates so tasks are verifiable (1 USD = x units).
_RATES = {"USD": 1.0, "KRW": 1380.0, "EUR": 0.92, "JPY": 151.0, "GBP": 0.79}


@tool
def convert_currency(amount: float, source: str, target: str) -> dict:
    """Convert an amount between currencies using today's reference rates.

    Args:
        amount: Amount in the source currency.
        source: ISO code of the source currency (USD, KRW, EUR, JPY, GBP).
        target: ISO code of the target currency.
    """
    s, t = source.upper().strip(), target.upper().strip()
    if s not in _RATES or t not in _RATES:
        raise ToolError(f"unsupported currency; supported: {sorted(_RATES)}")
    return {"amount": round(amount / _RATES[s] * _RATES[t], 2), "currency": t, "rate": round(_RATES[t] / _RATES[s], 6)}


@tool
def loan_payment(principal: float, annual_rate_pct: float, years: int) -> dict:
    """Monthly payment of a fixed-rate amortizing loan.

    Args:
        principal: Loan amount.
        annual_rate_pct: Annual interest rate in percent, e.g. 4.5.
        years: Loan term in years.
    """
    if principal <= 0 or years <= 0:
        raise ToolError("principal and years must be positive")
    n, r = years * 12, annual_rate_pct / 100 / 12
    pay = principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)
    return {"monthly_payment": round(pay, 2), "total_paid": round(pay * n, 2), "total_interest": round(pay * n - principal, 2)}


@tool
def compound_growth(initial: float, annual_rate_pct: float, years: int, monthly_contribution: float = 0.0) -> dict:
    """Future value of an investment with monthly compounding and optional monthly contributions.

    Args:
        initial: Starting balance.
        annual_rate_pct: Annual return in percent.
        years: Number of years.
        monthly_contribution: Amount added at the end of every month.
    """
    r, bal = annual_rate_pct / 100 / 12, float(initial)
    for _ in range(years * 12):
        bal = bal * (1 + r) + monthly_contribution
    return {"future_value": round(bal, 2)}
