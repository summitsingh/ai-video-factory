# YouTube upload setup

The uploader publishes finished pipeline runs to YouTube via the YouTube Data
API v3. It is an optional extra; the rest of the pipeline works without it.

## Install

```bash
uv sync --extra youtube
# or
pip install "ai-video-factory[youtube]"
```

## One-time Google setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **YouTube Data API v3** for the project.
3. Configure the OAuth consent screen (External, add yourself as a test user
   while the app is in testing mode).
4. Create an OAuth client ID of type **Desktop app** and download
   `client_secrets.json`.
5. Run the one-time auth flow and approve access for the Google account that
   owns the target channel:

```bash
ai-video-factory youtube auth --client-secrets /path/to/client_secrets.json
```

The refresh token is stored at `data/youtube/token.json` with owner-only
permissions and refreshes automatically. If the channel is a Brand Account,
authorize with the Google account that manages it.

## Usage

```bash
# Check auth state and today's quota usage
ai-video-factory youtube status

# Validate everything without calling the API
ai-video-factory youtube upload --run-id <run-id> --dry-run

# Upload a completed run (default privacy: unlisted)
ai-video-factory youtube upload --run-id <run-id>

# Go public only with the explicit confirmation flag
ai-video-factory youtube upload --run-id <run-id> --privacy public --confirm-public

# Or upload explicit files instead of a run
ai-video-factory youtube upload --video out.mp4 --title "My title" \
    --thumbnail thumb.png --captions subs.srt --tags "space,nasa"
```

Uploading from `--run-id` verifies the run manifest's SHA-256 digests before
upload, uses the pipeline's thumbnail, SRT captions, and chapter markers, and
defaults the title/description from the run inputs (overridable with flags).

## Quota

Each upload costs quota units against the project's daily budget (default
10,000, override with `AVF_YOUTUBE_QUOTA_BUDGET`):

- `videos.insert`: 1600
- `thumbnails.set`: 50
- `captions.insert`: 50

The uploader refuses to start when the remaining budget cannot cover the full
package, so a failed run never half-spends the day's quota.

## Safety rails

- Default privacy is **unlisted**. Publishing as public requires
  `--confirm-public`; there is no silent path to public.
- Resumable uploads retry 5xx and network errors with exponential backoff
  (6 attempts max); 4xx errors fail fast.
- A tampered or missing artifact (digest mismatch vs the run manifest)
  aborts the upload.

## Recommended rollout

Keep uploads unlisted for a 7-day soak, review each video, then enable the
daily public schedule with per-channel configuration and a review queue.
