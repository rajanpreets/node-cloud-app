import paypalrestsdk
from config import PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET

# Initialize PayPal SDK
paypalrestsdk.configure({
    "mode": "sandbox",  # Change to "live" in production
    "client_id": PAYPAL_CLIENT_ID,
    "client_secret": PAYPAL_CLIENT_SECRET
})

def is_subscribed(user_email):
    """
    Check if the user has an active PayPal subscription.
    """
    return True  # Implement PayPal subscription check
