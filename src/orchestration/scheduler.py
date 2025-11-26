"""
Pipeline Scheduler using APScheduler.

Manages scheduled jobs for:
- Daily drift checks (06:00 UTC)
- Weekly retraining (Sunday 02:00 UTC)
- Monthly hyperparameter optimization (1st Sunday 00:00 UTC)

Usage:
    scheduler = PipelineScheduler(pipeline)
    scheduler.start()  # Blocking
    
    # Or with custom schedule:
    scheduler.add_daily_job(my_func, hour=8, minute=30)
"""

from datetime import datetime
from typing import Callable, Optional, Dict, Any
import signal
import sys

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    print("Warning: APScheduler not installed. Install with: pip install apscheduler")


class PipelineScheduler:
    """
    Manages scheduled pipeline jobs using APScheduler.
    
    Default schedule:
    - Daily: 06:00 UTC - Drift check
    - Weekly: Sunday 02:00 UTC - Retrain evaluation
    - Monthly: 1st Sunday 00:00 UTC - Hyperparameter optimization
    """
    
    def __init__(
        self,
        pipeline: Optional[Any] = None,
        blocking: bool = True,
        timezone: str = "UTC",
    ):
        """
        Initialize scheduler.
        
        Args:
            pipeline: ContinuousPipeline instance (optional)
            blocking: Use blocking scheduler (True) or background (False)
            timezone: Timezone for cron expressions
        """
        if not APSCHEDULER_AVAILABLE:
            raise ImportError("APScheduler required. Install with: pip install apscheduler")
        
        self.pipeline = pipeline
        self.timezone = timezone
        
        if blocking:
            self.scheduler = BlockingScheduler(timezone=timezone)
        else:
            self.scheduler = BackgroundScheduler(timezone=timezone)
        
        # Add event listeners
        self.scheduler.add_listener(
            self._on_job_executed,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def shutdown(signum, frame):
            print("\nReceived shutdown signal. Stopping scheduler...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
    
    def _on_job_executed(self, event):
        """Log job execution events."""
        if event.exception:
            print(f"Job {event.job_id} failed: {event.exception}")
        else:
            print(f"Job {event.job_id} completed at {datetime.now()}")
    
    def add_daily_job(
        self,
        func: Callable,
        job_id: str = "daily_job",
        hour: int = 6,
        minute: int = 0,
        **kwargs,
    ) -> None:
        """
        Add a daily job.
        
        Args:
            func: Function to execute
            job_id: Unique job identifier
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            **kwargs: Additional arguments for the function
        """
        self.scheduler.add_job(
            func,
            CronTrigger(hour=hour, minute=minute, timezone=self.timezone),
            id=job_id,
            kwargs=kwargs,
            replace_existing=True,
        )
        print(f"Added daily job '{job_id}' at {hour:02d}:{minute:02d} {self.timezone}")
    
    def add_weekly_job(
        self,
        func: Callable,
        job_id: str = "weekly_job",
        day_of_week: str = "sun",
        hour: int = 2,
        minute: int = 0,
        **kwargs,
    ) -> None:
        """
        Add a weekly job.
        
        Args:
            func: Function to execute
            job_id: Unique job identifier
            day_of_week: Day to run (mon, tue, wed, thu, fri, sat, sun)
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            **kwargs: Additional arguments for the function
        """
        self.scheduler.add_job(
            func,
            CronTrigger(
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                timezone=self.timezone
            ),
            id=job_id,
            kwargs=kwargs,
            replace_existing=True,
        )
        print(f"Added weekly job '{job_id}' on {day_of_week} at {hour:02d}:{minute:02d} {self.timezone}")
    
    def add_monthly_job(
        self,
        func: Callable,
        job_id: str = "monthly_job",
        day: str = "1-7",  # First week
        day_of_week: str = "sun",  # First Sunday
        hour: int = 0,
        minute: int = 0,
        **kwargs,
    ) -> None:
        """
        Add a monthly job (runs on first Sunday of month by default).
        
        Args:
            func: Function to execute
            job_id: Unique job identifier
            day: Day of month (1-31, or range like "1-7")
            day_of_week: Day of week to run
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            **kwargs: Additional arguments for the function
        """
        self.scheduler.add_job(
            func,
            CronTrigger(
                day=day,
                day_of_week=day_of_week,
                hour=hour,
                minute=minute,
                timezone=self.timezone
            ),
            id=job_id,
            kwargs=kwargs,
            replace_existing=True,
        )
        print(f"Added monthly job '{job_id}' on day {day}, {day_of_week} at {hour:02d}:{minute:02d} {self.timezone}")
    
    def setup_default_schedule(self) -> None:
        """
        Setup default pipeline schedule:
        - Daily drift check at 06:00 UTC
        - Weekly retrain on Sunday at 02:00 UTC
        - Monthly hyperopt on 1st Sunday at 00:00 UTC
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not set. Provide pipeline in constructor.")
        
        # Daily drift check
        self.add_daily_job(
            self.pipeline.run_daily_drift_check,
            job_id="daily_drift_check",
            hour=6,
            minute=0,
        )
        
        # Weekly retrain
        self.add_weekly_job(
            self.pipeline.run_weekly_retrain,
            job_id="weekly_retrain",
            day_of_week="sun",
            hour=2,
            minute=0,
        )
        
        # Monthly hyperopt
        self.add_monthly_job(
            self.pipeline.run_monthly_hyperopt,
            job_id="monthly_hyperopt",
            day="1-7",
            day_of_week="sun",
            hour=0,
            minute=0,
        )
        
        print("\nDefault schedule configured:")
        self.print_jobs()
    
    def print_jobs(self) -> None:
        """Print all scheduled jobs."""
        jobs = self.scheduler.get_jobs()
        
        if not jobs:
            print("No jobs scheduled.")
            return
        
        print(f"\n{'='*60}")
        print("Scheduled Jobs")
        print(f"{'='*60}")
        
        for job in jobs:
            next_run = job.next_run_time
            if next_run:
                print(f"  {job.id}:")
                print(f"    Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                print(f"    Trigger: {job.trigger}")
            else:
                print(f"  {job.id}: (paused)")
        
        print(f"{'='*60}\n")
    
    def start(self) -> None:
        """Start the scheduler (blocking if configured)."""
        print(f"\nStarting scheduler at {datetime.now()}")
        print("Press Ctrl+C to stop.\n")
        
        self.print_jobs()
        self.scheduler.start()
    
    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        print("Stopping scheduler...")
        self.scheduler.shutdown(wait=True)
        print("Scheduler stopped.")
    
    def run_job_now(self, job_id: str) -> None:
        """Manually trigger a job immediately."""
        job = self.scheduler.get_job(job_id)
        if job:
            print(f"Running job '{job_id}' manually...")
            job.func(**job.kwargs if job.kwargs else {})
        else:
            print(f"Job '{job_id}' not found.")
    
    def pause_job(self, job_id: str) -> None:
        """Pause a scheduled job."""
        self.scheduler.pause_job(job_id)
        print(f"Job '{job_id}' paused.")
    
    def resume_job(self, job_id: str) -> None:
        """Resume a paused job."""
        self.scheduler.resume_job(job_id)
        print(f"Job '{job_id}' resumed.")
    
    def remove_job(self, job_id: str) -> None:
        """Remove a scheduled job."""
        self.scheduler.remove_job(job_id)
        print(f"Job '{job_id}' removed.")


def create_scheduler_from_config(
    config: Dict[str, Any],
    pipeline: Optional[Any] = None,
) -> PipelineScheduler:
    """
    Create scheduler from configuration dict.
    
    Example config:
    {
        "timezone": "UTC",
        "blocking": true,
        "jobs": {
            "daily_drift_check": {
                "enabled": true,
                "hour": 6,
                "minute": 0
            },
            "weekly_retrain": {
                "enabled": true,
                "day_of_week": "sun",
                "hour": 2
            }
        }
    }
    """
    scheduler = PipelineScheduler(
        pipeline=pipeline,
        blocking=config.get("blocking", True),
        timezone=config.get("timezone", "UTC"),
    )
    
    jobs_config = config.get("jobs", {})
    
    if pipeline:
        # Daily drift check
        daily_config = jobs_config.get("daily_drift_check", {})
        if daily_config.get("enabled", True):
            scheduler.add_daily_job(
                pipeline.run_daily_drift_check,
                job_id="daily_drift_check",
                hour=daily_config.get("hour", 6),
                minute=daily_config.get("minute", 0),
            )
        
        # Weekly retrain
        weekly_config = jobs_config.get("weekly_retrain", {})
        if weekly_config.get("enabled", True):
            scheduler.add_weekly_job(
                pipeline.run_weekly_retrain,
                job_id="weekly_retrain",
                day_of_week=weekly_config.get("day_of_week", "sun"),
                hour=weekly_config.get("hour", 2),
                minute=weekly_config.get("minute", 0),
            )
        
        # Monthly hyperopt
        monthly_config = jobs_config.get("monthly_hyperopt", {})
        if monthly_config.get("enabled", True):
            scheduler.add_monthly_job(
                pipeline.run_monthly_hyperopt,
                job_id="monthly_hyperopt",
                day=monthly_config.get("day", "1-7"),
                day_of_week=monthly_config.get("day_of_week", "sun"),
                hour=monthly_config.get("hour", 0),
                minute=monthly_config.get("minute", 0),
            )
    
    return scheduler


if __name__ == "__main__":
    # Demo scheduler with dummy jobs
    def dummy_daily():
        print(f"[{datetime.now()}] Daily job executed!")
    
    def dummy_weekly():
        print(f"[{datetime.now()}] Weekly job executed!")
    
    scheduler = PipelineScheduler(blocking=False)
    
    # Add demo jobs (run every minute for testing)
    scheduler.scheduler.add_job(
        dummy_daily,
        CronTrigger(minute="*/1"),  # Every minute
        id="demo_daily",
    )
    
    print("Demo scheduler with 1-minute interval job.")
    print("Press Ctrl+C to stop.\n")
    
    scheduler.print_jobs()
    
    # Run for a short time
    import time
    scheduler.scheduler.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop()

