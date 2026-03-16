from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from .drift_detector import DriftResult

logger = logging.getLogger("monitoring.alerts")


@dataclass
class Alert:
    timestamp: str
    level: str  # "WARNING" or "CRITICAL"
    message: str
    feature: str
    test: str
    score: float


class Alerter:
    """Rule-based alerter that checks drift scores against configured thresholds."""

    def __init__(
        self,
        psi_warning: float = 0.1,
        psi_critical: float = 0.25,
        ks_p_threshold: float = 0.05,
    ) -> None:
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.ks_p_threshold = ks_p_threshold
        self._alert_history: list[Alert] = []

    def check(self, drift_results: list[DriftResult]) -> list[Alert]:
        """Evaluate drift results against thresholds and fire alerts.

        Returns:
            List of new Alert objects generated.
        """
        new_alerts = []
        for result in drift_results:
            alert = self._evaluate(result)
            if alert:
                new_alerts.append(alert)
                self._alert_history.append(alert)
                self.notify(alert)
        return new_alerts

    def _evaluate(self, result: DriftResult) -> Alert | None:
        ts = datetime.now().isoformat()
        if result.test == "psi":
            if result.score > self.psi_critical:
                return Alert(ts, "CRITICAL", f"PSI={result.score:.3f} > {self.psi_critical} on '{result.feature}'", result.feature, result.test, result.score)
            elif result.score > self.psi_warning:
                return Alert(ts, "WARNING", f"PSI={result.score:.3f} > {self.psi_warning} on '{result.feature}'", result.feature, result.test, result.score)
        elif result.test == "ks" and result.p_value is not None:
            if result.p_value < self.ks_p_threshold:
                return Alert(ts, "WARNING", f"KS p={result.p_value:.4f} < {self.ks_p_threshold} on '{result.feature}'", result.feature, result.test, result.score)
        elif result.test == "chi2" and result.p_value is not None:
            if result.p_value < self.ks_p_threshold:
                return Alert(ts, "WARNING", f"Chi2 p={result.p_value:.4f}: prediction distribution shifted", result.feature, result.test, result.score)
        elif result.test == "mmd" and result.is_drift:
            return Alert(ts, "WARNING", f"MMD={result.score:.4f}: embedding drift detected", result.feature, result.test, result.score)
        return None

    def notify(self, alert: Alert) -> None:
        """Dispatch alert. Override subclasses for email/Slack."""
        self._log_alert(alert)

    def _log_alert(self, alert: Alert) -> None:
        msg = f"[{alert.level}] {alert.timestamp} | {alert.message}"
        if alert.level == "CRITICAL":
            logger.error(msg)
        else:
            logger.warning(msg)
        print(msg)

    @property
    def alert_history(self) -> list[Alert]:
        return self._alert_history


class SlackAlerter(Alerter):
    """Stub for Slack alerting. Implement webhook_url for production."""

    def __init__(self, webhook_url: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.webhook_url = webhook_url

    def notify(self, alert: Alert) -> None:
        self._log_alert(alert)
        if self.webhook_url:
            # Real implementation: requests.post(self.webhook_url, json={"text": alert.message})
            print(f"[Slack stub] Would send: {alert.message}")
