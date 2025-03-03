import requests
import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class PayPalClient:
    def __init__(self):
        self.client_id = st.secrets["PAYPAL_CLIENT_ID"]
        self.client_secret = st.secrets["PAYPAL_SECRET"]
        self.base_url = "https://api-m.sandbox.paypal.com" if st.secrets.get("PAYPAL_SANDBOX", True) \
                       else "https://api-m.paypal.com"
        self.access_token = None
        self.token_expiry = None
        
    def _get_auth_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_access_token()}"
        }
        
    def get_access_token(self) -> str:
        """Get or refresh access token"""
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token
            
        try:
            response = requests.post(
                f"{self.base_url}/v1/oauth2/token",
                auth=(self.client_id, self.client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
                timeout=10
            )
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.token_expiry = datetime.now() + timedelta(seconds=token_data["expires_in"] - 60)
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to get access token: {str(e)}")
            raise
        
    def create_order(self, amount: float) -> Optional[Dict]:
        """Create PayPal payment order"""
        try:
            headers = self._get_auth_headers()
            payload = {
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {
                        "currency_code": "USD",
                        "value": f"{amount:.2f}"
                    }
                }],
                "payment_source": {
                    "paypal": {
                        "experience_context": {
                            "return_url": st.secrets["PAYPAL_RETURN_URL"],
                            "cancel_url": st.secrets["PAYPAL_CANCEL_URL"]
                        }
                    }
                }
            }
            
            response = requests.post(
                f"{self.base_url}/v2/checkout/orders",
                json=payload,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Order creation failed: {str(e)}")
            return None
        
    def verify_payment(self, order_id: str) -> bool:
        """Verify payment completion"""
        try:
            headers = self._get_auth_headers()
            response = requests.get(
                f"{self.base_url}/v2/checkout/orders/{order_id}",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "COMPLETED" and \
                   float(data["purchase_units"][0]["amount"]["value"]) >= 10.00
        except Exception as e:
            logger.error(f"Payment verification failed: {str(e)}")
            return False
