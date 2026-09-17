# Channel scheduler: automated daily publishing

The scheduler runs dedicated-topic channels end to end: research a fresh
topic, render a long-form documentary, upload it **unlisted**, and file it
in a review queue. Nothing goes public without human approval.

## Channel config

`config/channels.toml` defines each channel:

```toml
[[channels]]
name = "Deep Space Diaries"
slug = "deep-space-diaries"
topic_focus = "space exploration documentaries"
theme = "space"
target_duration_minutes = 25.0
schedule_time = "09:00"
timezone = "America/Chicago"
privacy = "unlisted"
category_id = "28"
tags = ["space documentary", "astronomy"]
language = "en"
research_queries = ["space exploration", "astronomy discoveries"]
trend_source = "all"
```

Two example channels ship in `config/channels.toml`; edit or replace them.

## Topic deduplication

`data/youtube/topics.jsonl` is the append-only ledger of covered topics
per channel. A candidate is rejected when its normalized title matches
exactly or is a near-duplicate (token-set Jaccard >= 0.85) of anything the
channel already covered. A failed pipeline run does **not** burn its
topic: the ledger entry is written only after a successful upload, so the
same topic is retried the next day.

## At-most-one runner

`scheduler run` takes an exclusive file lock (`data/youtube/scheduler.lock`).
A second invocation while one is running exits with code 3 instead of
doubling up renders or uploads.

## Commands

```bash
# See what is due
ai-video-factory scheduler due

# Monitoring snapshot (last run, due state, topics used, pending reviews)
ai-video-factory scheduler status

# Run all due channels (research, render, unlisted upload, review queue)
ai-video-factory scheduler run

# Preview what would run without rendering or uploading
ai-video-factory scheduler run --dry-run

# Force one channel regardless of due state
ai-video-factory scheduler run --channel deep-space-diaries --force
```

## Review queue

Uploads land unlisted and appear in `review list`:

```bash
ai-video-factory review list
ai-video-factory review approve <video-id>            # approve, stays unlisted
ai-video-factory review approve <video-id> --publish  # approve AND go public
ai-video-factory review reject <video-id>             # keep unlisted, record decision
```

## Daily automation

Run `scheduler run` from cron once or twice a day (after the latest
channel's scheduled time). Example crontab (09:30 America/Chicago):

```
30 9 * * * cd /path/to/ai-video-factory && uv run ai-video-factory scheduler run >> data/youtube/scheduler.log 2>&1
```

Each channel runs at most once per calendar day in its own timezone.
Check `scheduler status` for failures; failed runs record their error in
`data/youtube/state/<slug>.json`.

## Soak recommendation

Before turning on daily public publishing, run the scheduler for a week
with uploads staying unlisted. Review every video, then publish the
keepers with `review approve --publish`.
