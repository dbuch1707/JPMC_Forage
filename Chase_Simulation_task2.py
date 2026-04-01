#!/usr/bin/env python
# coding: utf-8

# In[1]:


def price_storage_contract(injection_dates, withdrawal_dates, injection_rate, 
                           withdrawal_rate, max_volume, storage_cost_per_month):
    """
    Calculates the net value of a gas storage contract.
    """
    total_value = 0
    current_volume = 0
    
    # 1. Handle Injections (Buying Gas)
    print("--- Injection Phase ---")
    for date in injection_dates:
        # Get price from your previous estimate_price function
        price = get_price_for_date(date) 
        
        # Determine how much we can actually inject
        space_available = max_volume - current_volume
        amount_to_inject = min(injection_rate, space_available)
        
        cost = amount_to_inject * price
        total_value -= cost
        current_volume += amount_to_inject
        
        print(f"Date: {date} | Injected: {amount_to_inject} units | Cost: ${cost:,.2f}")

    # 2. Handle Withdrawals (Selling Gas)
    print("\n--- Withdrawal Phase ---")
    for date in withdrawal_dates:
        price = get_price_for_date(date)
        
        # Determine how much we can actually withdraw
        amount_to_withdraw = min(withdrawal_rate, current_volume)
        
        revenue = amount_to_withdraw * price
        total_value += revenue
        current_volume -= amount_to_withdraw
        
        print(f"Date: {date} | Withdrawn: {amount_to_withdraw} units | Revenue: ${revenue:,.2f}")

    # 3. Deduct Storage Costs
    # Assuming storage cost is paid on the max capacity for the duration of the contract
    # (Simplified for the prototype)
    months_in_storage = len(injection_dates) + len(withdrawal_dates)
    total_storage_fees = max_volume * storage_cost_per_month * months_in_storage
    total_value -= total_storage_fees
    
    print(f"\nTotal Storage Fees: ${total_storage_fees:,.2f}")
    print(f"Final Contract Value: ${total_value:,.2f}")
    
    return total_value

# --- Sample Test Inputs ---
inj_dates = ["2024-10-31", "2024-11-30"] # Buying when cheaper (Autumn)
with_dates = ["2025-01-31", "2025-02-28"] # Selling when expensive (Winter)

contract_value = price_storage_contract(
    injection_dates=inj_dates,
    withdrawal_dates=with_dates,
    injection_rate=50000, 
    withdrawal_rate=50000,
    max_volume=100000,
    storage_cost_per_month=1000
)


# In[ ]:




