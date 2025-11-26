"""
Notification Service for Continuous Pipeline.

Sends alerts for:
- Critical drift detected
- Retrain completed
- Model deployed
- Performance degradation

Supports:
- Slack webhooks
- Email (via SMTP)
- Console output (for testing)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os


@dataclass
class Alert:
    """Alert message structure."""
    title: str
    message: str
    severity: str  # "critical", "warning", "info"
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class Notifier(ABC):
    """Abstract base class for notification services."""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert. Returns True if successful."""
        pass
    
    def send_critical(self, title: str, message: str, **metadata) -> bool:
        """Convenience method for critical alerts."""
        return self.send(Alert(
            title=title,
            message=message,
            severity="critical",
            timestamp=datetime.now(),
            metadata=metadata,
        ))
    
    def send_warning(self, title: str, message: str, **metadata) -> bool:
        """Convenience method for warning alerts."""
        return self.send(Alert(
            title=title,
            message=message,
            severity="warning",
            timestamp=datetime.now(),
            metadata=metadata,
        ))
    
    def send_info(self, title: str, message: str, **metadata) -> bool:
        """Convenience method for info alerts."""
        return self.send(Alert(
            title=title,
            message=message,
            severity="info",
            timestamp=datetime.now(),
            metadata=metadata,
        ))


class ConsoleNotifier(Notifier):
    """Print alerts to console (for testing/development)."""
    
    def send(self, alert: Alert) -> bool:
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟡",
            "info": "🟢",
        }.get(alert.severity, "⚪")
        
        print(f"\n{'='*60}")
        print(f"{severity_emoji} [{alert.severity.upper()}] {alert.title}")
        print(f"{'='*60}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{alert.message}")
        
        if alert.metadata:
            print(f"\nMetadata:")
            for k, v in alert.metadata.items():
                print(f"  {k}: {v}")
        
        print(f"{'='*60}\n")
        return True


class SlackNotifier(Notifier):
    """Send alerts to Slack via webhook."""
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        channel: Optional[str] = None,
        username: str = "Fraud Detection Bot",
    ):
        """
        Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL (or set SLACK_WEBHOOK_URL env var)
            channel: Override channel (optional)
            username: Bot username
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel
        self.username = username
        
        if not self.webhook_url:
            print("Warning: Slack webhook URL not configured. Alerts will be logged only.")
    
    def send(self, alert: Alert) -> bool:
        if not self.webhook_url:
            # Fallback to console
            return ConsoleNotifier().send(alert)
        
        try:
            import requests
        except ImportError:
            print("requests library not installed. Install with: pip install requests")
            return ConsoleNotifier().send(alert)
        
        color = {
            "critical": "#FF0000",
            "warning": "#FFA500",
            "info": "#36A64F",
        }.get(alert.severity, "#808080")
        
        emoji = {
            "critical": ":rotating_light:",
            "warning": ":warning:",
            "info": ":white_check_mark:",
        }.get(alert.severity, ":bell:")
        
        # Build Slack message
        payload = {
            "username": self.username,
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.upper(),
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True,
                        },
                    ],
                    "footer": "Fraud Detection Pipeline",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ],
        }
        
        # Add metadata fields
        if alert.metadata:
            for k, v in list(alert.metadata.items())[:5]:  # Limit to 5 fields
                payload["attachments"][0]["fields"].append({
                    "title": k,
                    "value": str(v),
                    "short": True,
                })
        
        if self.channel:
            payload["channel"] = self.channel
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            
            if response.status_code == 200:
                return True
            else:
                print(f"Slack API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False


class EmailNotifier(Notifier):
    """Send alerts via email (SMTP)."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None,
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_host: SMTP server host (or SMTP_HOST env var)
            smtp_port: SMTP server port
            smtp_user: SMTP username (or SMTP_USER env var)
            smtp_password: SMTP password (or SMTP_PASSWORD env var)
            from_email: From email address
            to_emails: List of recipient emails
        """
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("ALERT_FROM_EMAIL", "alerts@example.com")
        self.to_emails = to_emails or os.getenv("ALERT_TO_EMAILS", "").split(",")
        
        if not self.smtp_host:
            print("Warning: SMTP not configured. Email alerts disabled.")
    
    def send(self, alert: Alert) -> bool:
        if not self.smtp_host or not self.to_emails:
            return ConsoleNotifier().send(alert)
        
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
        msg["From"] = self.from_email
        msg["To"] = ", ".join(self.to_emails)
        
        # Plain text version
        text = f"""
{alert.title}
{'='*50}

Severity: {alert.severity.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """
        
        # HTML version
        html = f"""
<html>
<body>
<h2 style="color: {'red' if alert.severity == 'critical' else 'orange' if alert.severity == 'warning' else 'green'}">
{alert.title}
</h2>
<p><strong>Severity:</strong> {alert.severity.upper()}</p>
<p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
<hr>
<p>{alert.message}</p>
<hr>
<pre>{json.dumps(alert.metadata, indent=2)}</pre>
</body>
</html>
        """
        
        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            return True
            
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False


class CompositeNotifier(Notifier):
    """Send to multiple notification channels."""
    
    def __init__(self, notifiers: List[Notifier]):
        self.notifiers = notifiers
    
    def send(self, alert: Alert) -> bool:
        results = []
        for notifier in self.notifiers:
            try:
                results.append(notifier.send(alert))
            except Exception as e:
                print(f"Notifier {type(notifier).__name__} failed: {e}")
                results.append(False)
        
        return any(results)  # Success if at least one succeeded


def create_notifier_from_config(config: Dict) -> Notifier:
    """
    Create notifier from configuration dict.
    
    Example config:
    {
        "type": "composite",
        "channels": [
            {"type": "console"},
            {"type": "slack", "webhook_url": "..."},
            {"type": "email", "smtp_host": "...", "to_emails": ["..."]}
        ]
    }
    """
    notifier_type = config.get("type", "console")
    
    if notifier_type == "console":
        return ConsoleNotifier()
    
    elif notifier_type == "slack":
        return SlackNotifier(
            webhook_url=config.get("webhook_url"),
            channel=config.get("channel"),
        )
    
    elif notifier_type == "email":
        return EmailNotifier(
            smtp_host=config.get("smtp_host"),
            smtp_port=config.get("smtp_port", 587),
            smtp_user=config.get("smtp_user"),
            smtp_password=config.get("smtp_password"),
            from_email=config.get("from_email"),
            to_emails=config.get("to_emails"),
        )
    
    elif notifier_type == "composite":
        channels = config.get("channels", [])
        return CompositeNotifier([
            create_notifier_from_config(ch) for ch in channels
        ])
    
    else:
        print(f"Unknown notifier type: {notifier_type}, falling back to console")
        return ConsoleNotifier()


if __name__ == "__main__":
    # Test notifications
    notifier = ConsoleNotifier()
    
    notifier.send_critical(
        "Critical Drift Detected",
        "3 features have PSI > 0.25: account_age_days, log_price, rooms",
        drifted_features=3,
        max_psi=0.35,
    )
    
    notifier.send_warning(
        "Weekly Retrain Complete",
        "New model trained with AUC-PR 0.7234 (+1.2% over production)",
        new_auc_pr=0.7234,
        improvement_pct=1.2,
    )
    
    notifier.send_info(
        "Model Deployed",
        "Model v3 promoted to Production stage",
        model_version=3,
        previous_version=2,
    )

