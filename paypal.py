import paypalrestsdk
import streamlit as st
from datetime import datetime
from database import store_subscription

paypalrestsdk.configure({
    "mode": "sandbox",  # "live" for production
    "client_id": st.secrets["PAYPAL_CLIENT_ID"],
    "client_secret": st.secrets["PAYPAL_SECRET"]
})

def create_subscription(user_id: int, email: str):
    try:
        subscription = paypalrestsdk.Subscription({
            "plan_id": st.secrets["PAYPAL_PLAN_ID"],
            "start_time": datetime.utcnow().isoformat() + "Z",
            "subscriber": {
                "email_address": email
            },
            "application_context": {
                "return_url": st.secrets["PAYPAL_RETURN_URL"],
                "cancel_url": st.secrets["PAYPAL_CANCEL_URL"]
            }
        })
        
        if subscription.create():
            store_subscription(user_id, {
                "id": subscription.id,
                "plan_id": st.secrets["PAYPAL_PLAN_ID"],
                "status": subscription.status,
                "start_time": subscription.start_time,
                "subscriber": subscription.subscriber
            })
            return subscription.id
        return None
    except Exception as e:
        st.error(f"Subscription creation failed: {str(e)}")
        return None

def verify_subscription(subscription_id: str):
    try:
        subscription = paypalrestsdk.Subscription.find(subscription_id)
        return subscription.status == "ACTIVE"
    except Exception as e:
        st.error(f"Subscription verification failed: {str(e)}")
        return False

def handle_webhook(event_data: dict):
    try:
        if not paypalrestsdk.WebhookEvent.verify(
            event_data['transmission_id'],
            event_data['transmission_time'],
            event_data['transmission_sig'],
            event_data['webhook_id'],
            st.secrets["PAYPAL_WEBHOOK_ID"]
        ):
            return False

        event_type = event_data['event_type']
        resource = event_data['resource']
        
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            # Handle subscription activation
            pass
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            # Handle cancellation
            pass
        
        return True
    except Exception as e:
        st.error(f"Webhook handling failed: {str(e)}")
        return False
